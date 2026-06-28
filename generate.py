"""
generate.py — sample text from a trained checkpoint.

Examples:
    python generate.py
    python generate.py --prompt "ROMEO:" --tokens 400 --temperature 0.8 --top_k 20

Train a checkpoint first with:  python -m gpt.train
"""
import argparse
import os
import torch

from gpt.config import GPTConfig
from gpt.model import GPTLanguageModel
from gpt.train import get_device
from gpt.sample import generate_text


def main():
    p = argparse.ArgumentParser(description="Generate Shakespeare from a trained GPT.")
    p.add_argument("--prompt", default="\n", help="seed text to continue")
    p.add_argument("--tokens", type=int, default=500, help="number of characters to generate")
    p.add_argument("--temperature", type=float, default=1.0, help="<1 safer, >1 more creative")
    p.add_argument("--top_k", type=int, default=None, help="restrict sampling to the k most likely chars")
    p.add_argument("--checkpoint", default="shakespeare_gpt.pt", help="path to the saved weights")
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    text = open(os.path.join(here, "input.txt")).read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}

    device = get_device()
    model = GPTLanguageModel(GPTConfig(vocab_size=len(chars))).to(device)
    model.load_state_dict(torch.load(os.path.join(here, args.checkpoint), map_location=device))

    print(generate_text(
        model, stoi, itos,
        prompt=args.prompt, max_new_tokens=args.tokens,
        temperature=args.temperature, top_k=args.top_k,
    ))


if __name__ == "__main__":
    main()
