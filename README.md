# shakespeare-gpt

A small GPT-style language model built **from scratch in raw PyTorch** — no
HuggingFace, no transformer libraries — and trained on Shakespeare. Every
component (tokenizer, embeddings, self-attention, transformer blocks, training
loop, sampler) is implemented from first principles.

The model is a ~0.82M-parameter character-level transformer. Trained on a ~98K-
character slice of tiny-shakespeare, it learns the format and texture of the
text (speaker names, dialogue, line breaks, Shakespearean word-shapes).

## Architecture

```
ids ──▶ token + position embedding ──▶ N× Transformer Block ──▶ LayerNorm ──▶ Linear head ──▶ logits
                                          │
                                          ├── multi-head causal self-attention   (tokens communicate)
                                          └── feed-forward network               (per-token compute)
                                          each wrapped in a pre-norm residual
```

Defaults (`gpt/config.py`): `block_size=128`, `n_embd=128`, `n_head=4`,
`n_layer=4`, `dropout=0.1`.

## Layout

```
shakespeare-gpt/
├── gpt/
│   ├── config.py        # GPTConfig: all hyperparameters
│   ├── tokenizer.py     # character-level tokenizer (text <-> ids)
│   ├── embeddings.py    # token + positional embeddings
│   ├── attention.py     # Head + MultiHeadAttention (Q/K/V, scaling, causal mask)
│   ├── block.py         # FeedForward + Block (residuals + layer norm)
│   ├── model.py         # GPTLanguageModel (assembles the full network)
│   ├── train.py         # cross-entropy + AdamW training loop
│   └── sample.py        # autoregressive generation (temperature, top-k, greedy)
├── generate.py          # CLI: sample from a trained checkpoint
├── input.txt            # training corpus (~98K chars of tiny-shakespeare)
├── assets/              # figures explaining each component
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate   # Python 3.11 or 3.12 recommended
pip install -r requirements.txt
```

## Train

```bash
python -m gpt.train          # trains, prints train/val loss, saves shakespeare_gpt.pt
```

Auto-selects CUDA / Apple-Silicon `mps` / CPU. Initial loss should be ≈ `ln(61) ≈ 4.11`
(a uniform guess over the 61-character vocabulary) — a quick sanity check that
the loss is wired up correctly.

## Generate

```bash
python generate.py --prompt "ROMEO:" --tokens 400 --temperature 0.8 --top_k 20
```

## Notes

- **Overfitting is expected and visible.** With ~0.82M parameters and only ~98K
  characters, training loss keeps falling while validation loss bottoms out
  (around 1.59) and then rises — the model memorizing rather than generalizing.
  The single biggest quality win is more data: drop the full ~1.1M-character
  tiny-shakespeare in as `input.txt` and (optionally) enlarge the model.
- Inspired by Andrej Karpathy's nanoGPT / "Let's build GPT" — reimplemented from
  scratch as a learning project.

## Figures

| Self-attention weights (causal) | Temperature reshapes sampling |
|---|---|
| ![attention](assets/attention_matrix.png) | ![temperature](assets/temperature_effect.png) |
