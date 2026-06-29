"""
generate.py — sample text from a trained checkpoint.

Examples:
    python generate.py
    python generate.py --prompt "ROMEO:" --tokens 400 --temperature 0.8 --top_k 20
    python generate.py --top_p 0.9

Train a checkpoint first with:  python -m gpt.train      (baseline)
                            or:  python experiment.py     (scaled-up, recommended)
"""
import argparse
import os
import torch

from gpt.config import GPTConfig
from gpt.model import GPTLanguageModel
from gpt.train import get_device
from gpt.sample import generate_text


def load_model(checkpoint_path, device):
    """Load either format:
      * self-describing dict from experiment.py: {model, config, stoi, itos}
      * bare state_dict from gpt.train: weights only (use default config + input.txt)
    Returns (model, stoi, itos).
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    here = os.path.dirname(os.path.abspath(__file__))

    if isinstance(ckpt, dict) and "model" in ckpt:
        config = GPTConfig(**ckpt["config"])
        model = GPTLanguageModel(config).to(device)
        model.lm_head.weight = model.token_position.token_emb.weight  # match weight tying
        model.load_state_dict(ckpt["model"])
        return model, ckpt["stoi"], ckpt["itos"]

    # bare state_dict -> rebuild the tokenizer from input.txt and use defaults
    text = open(os.path.join(here, "input.txt")).read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    model = GPTLanguageModel(GPTConfig(vocab_size=len(chars))).to(device)
    model.load_state_dict(ckpt)
    return model, stoi, itos


def main():
    p = argparse.ArgumentParser(description="Generate Shakespeare from a trained GPT.")
    p.add_argument("--prompt", default="\n", help="seed text to continue")
    p.add_argument("--tokens", type=int, default=500, help="number of characters to generate")
    p.add_argument("--temperature", type=float, default=1.0, help="<1 safer, >1 more creative")
    p.add_argument("--top_k", type=int, default=None, help="sample from the k most likely chars")
    p.add_argument("--top_p", type=float, default=None, help="nucleus sampling threshold (e.g. 0.9)")
    p.add_argument("--checkpoint", default="shakespeare_gpt.pt", help="path to the saved weights")
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    device = get_device()
    model, stoi, itos = load_model(os.path.join(here, args.checkpoint), device)

    print(generate_text(
        model, stoi, itos,
        prompt=args.prompt, max_new_tokens=args.tokens,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
    ))


if __name__ == "__main__":
    main()
