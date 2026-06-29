"""
experiment.py
=============
The "scale-up" run: train a stronger GPT and auto-report metrics (including a
ready-to-paste CV line).

Compared with the baseline (char model, ~0.8M params, ~98K chars) this script:
  * trains on the FULL ~1.1M-character corpus (~11x more data)
  * uses a larger model (default ~10.7M params; configurable)
  * adds weight tying, LR warmup + cosine decay, and gradient clipping
  * uses EARLY STOPPING: saves the checkpoint at the BEST validation loss
  * reports validation loss, perplexity, % reduction vs the random baseline,
    throughput, and prints a CV bullet with the real numbers filled in

ONE-TIME: get the full corpus (the included input.txt is only a ~98K slice):
    curl -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

THEN:
    python experiment.py                 # large model on whatever input.txt holds
    python experiment.py --small         # baseline config, for an apples-to-apples compare

On Apple Silicon this auto-selects the `mps` GPU. The large run is a few tens of
minutes; reduce --iters or use --small if you want it faster.
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from gpt.config import GPTConfig
from gpt.model import GPTLanguageModel
from gpt.train import get_device, get_batch, estimate_loss


def build_config(vocab_size, large=True):
    """Two presets. 'large' is the canonical char-GPT size (~10.7M params)."""
    if large:
        return GPTConfig(vocab_size=vocab_size, block_size=256, n_embd=384,
                         n_head=6, n_layer=6, dropout=0.2)
    return GPTConfig(vocab_size=vocab_size, block_size=128, n_embd=128,
                     n_head=4, n_layer=4, dropout=0.1)


def lr_at(step, warmup, max_steps, lr, min_lr):
    """Learning-rate schedule: warmup first, then cosine decay.

    Warmup:
    At the start of training the model weights are still random, so full-size
    updates can be unstable. For the first `warmup` steps, start with a tiny
    learning rate and increase it linearly until it reaches `lr`.

    Cosine decay:
    After warmup, gradually lower the learning rate from `lr` to `min_lr`.
    This lets training make large useful moves early, then smaller fine-tuning
    moves near the end.
    """
    if step < warmup:
        return lr * (step + 1) / warmup
    if step > max_steps:
        return min_lr
    ratio = (step - warmup) / max(1, max_steps - warmup)          # 0 -> 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))               # 1 -> 0
    return min_lr + coeff * (lr - min_lr)


def main():
    ap = argparse.ArgumentParser(description="Scale-up training run with metrics reporting.")
    ap.add_argument("--data", default="input.txt")
    ap.add_argument("--small", action="store_true", help="use the baseline config instead of large")
    # optional architecture overrides (to fit your hardware / time budget)
    ap.add_argument("--n_embd", type=int, default=None)
    ap.add_argument("--n_head", type=int, default=None)
    ap.add_argument("--n_layer", type=int, default=None)
    ap.add_argument("--block_size", type=int, default=None)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval_interval", type=int, default=250)
    ap.add_argument("--eval_iters", type=int, default=200)
    ap.add_argument("--out", default="shakespeare_gpt.pt")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--rest", type=float, default=0.0,
                    help="seconds to pause between steps — lowers average load/heat (slower, cooler)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    here = os.path.dirname(os.path.abspath(__file__))

    # --- data ---
    text = open(os.path.join(here, args.data)).read()
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    splits = {"train": data[:n], "val": data[n:]}

    # --- model (+ weight tying) ---
    config = build_config(vocab_size, large=not args.small)
    # apply any explicit architecture overrides from the CLI
    for field in ("n_embd", "n_head", "n_layer", "block_size"):
        val = getattr(args, field)
        if val is not None:
            setattr(config, field, val)
    model = GPTLanguageModel(config).to(device)
    # Weight tying: share the token-embedding table with the output head (a GPT-2
    # trick). They have the same shape (vocab_size, n_embd); sharing them saves
    # parameters and usually improves loss slightly.
    model.lm_head.weight = model.token_position.token_emb.weight

    n_params = sum(p.numel() for p in model.parameters())
    random_loss = math.log(vocab_size)
    print(f"device={device}  params={n_params:,}  chars={len(text):,}  vocab={vocab_size}")
    print(f"random-baseline loss = ln({vocab_size}) = {random_loss:.4f}  (perplexity {vocab_size})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    warmup = max(100, args.iters // 50)

    # --- training loop with early stopping ---
    best_val, best_iter, history = float("inf"), 0, []
    t0, tokens_seen = time.time(), 0
    for it in range(args.iters + 1):
        lr = lr_at(it, warmup, args.iters, args.lr, args.lr / 10)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if it % args.eval_interval == 0 or it == args.iters:
            losses = estimate_loss(model, splits, config.block_size, args.batch_size, args.eval_iters, device)
            history.append((it, losses["train"], losses["val"]))
            if losses["val"] < best_val:                       # early stopping: keep the best
                best_val, best_iter = losses["val"], it
                torch.save({"model": model.state_dict(), "config": vars(config),
                            "stoi": stoi, "itos": itos, "val_loss": best_val},
                           os.path.join(here, args.out))
            print(f"iter {it:5d} | lr {lr:.2e} | train {losses['train']:.4f} | "
                  f"val {losses['val']:.4f} | best {best_val:.4f}")

        xb, yb = get_batch(splits["train"], config.block_size, args.batch_size, device)
        logits = model(xb)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), yb.view(B * T))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Gradient clipping: if the combined gradient size is above 1.0,
        # scale all gradients down before optimizer.step() uses them.
        # This keeps one unusually large update from destabilizing training.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        # Count how many token positions were trained on this step:
        # B sequences * T positions per sequence.
        tokens_seen += B * T
        if args.rest:
            time.sleep(args.rest)   # duty-cycle: let the chip cool between steps
    dt = time.time() - t0

    # --- metrics ---
    ppl_random, ppl_best = math.exp(random_loss), math.exp(best_val)
    loss_reduction = 100 * (random_loss - best_val) / random_loss
    ppl_reduction = 100 * (ppl_random - ppl_best) / ppl_random
    print("\n==================== RESULTS ====================")
    print(f"best val loss   : {best_val:.4f}   (iter {best_iter}; early-stopped checkpoint saved to {args.out})")
    print(f"perplexity      : {ppl_best:.2f}    (random baseline {ppl_random:.0f})")
    print(f"loss reduction  : {loss_reduction:.0f}% below the random-token baseline")
    print(f"perplexity drop : {ppl_reduction:.0f}%  ({ppl_random:.0f} -> {ppl_best:.1f})")
    print(f"parameters      : {n_params:,}")
    print(f"throughput      : {tokens_seen / dt:,.0f} tokens/sec on {device}  ({dt:.0f}s total)")
    print("\n---------- CV bullet (numbers from THIS run) ----------")
    print(f"Trained a {n_params/1e6:.0f}M-parameter character-level transformer on "
          f"{len(text)/1e6:.1f}M characters with AdamW + cosine LR schedule, weight tying,\n"
          f"dropout and early stopping, reaching validation perplexity {ppl_best:.1f} "
          f"({loss_reduction:.0f}% cross-entropy reduction vs. the uniform-token baseline).")

    # --- loss-curve figure (optional; needs matplotlib) ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        it_, tr_, va_ = zip(*history)
        plt.figure(figsize=(7, 4.4))
        plt.plot(it_, tr_, label="train")
        plt.plot(it_, va_, label="val")
        plt.axhline(random_loss, ls="--", color="gray", label="random baseline")
        plt.scatter([best_iter], [best_val], color="red", zorder=5, label=f"best val {best_val:.2f}")
        plt.xlabel("iteration"); plt.ylabel("cross-entropy loss"); plt.legend()
        plt.title(f"Training curve ({n_params/1e6:.1f}M params, {len(text)/1e6:.1f}M chars)")
        plt.tight_layout()
        os.makedirs(os.path.join(here, "assets"), exist_ok=True)
        plt.savefig(os.path.join(here, "assets", "training_curve_full.png"), dpi=110)
        print("\nsaved assets/training_curve_full.png")
    except Exception as e:
        print("(figure skipped:", e, ")")

    # --- machine-readable metrics ---
    json.dump({"best_val_loss": best_val, "perplexity": ppl_best,
               "loss_reduction_pct": loss_reduction, "params": n_params,
               "chars": len(text), "best_iter": best_iter,
               "tokens_per_sec": tokens_seen / dt},
              open(os.path.join(here, "metrics.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
