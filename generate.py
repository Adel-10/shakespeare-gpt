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
    """Load a trained model checkpoint from a .pt file.

    A .pt file is a PyTorch-saved checkpoint. In this project it stores the
    learned weights from training, and sometimes extra metadata too.

    This function supports two formats:
      * self-describing dict from experiment.py: {model, config, stoi, itos}
      * bare state_dict from gpt.train: weights only (use default config + input.txt)

    Loading works by rebuilding a fresh model object first, then copying the
    saved trained weights into it. The checkpoint stores the learned tensors;
    the Python model structure still has to be recreated before those tensors
    can be used for generation.

    Returns (model, stoi, itos).
    """
    # Load the checkpoint tensors onto the device we will generate on.
    # map_location lets a checkpoint trained on one device load on another,
    # for example loading GPU-trained weights onto CPU or MPS.
    ckpt = torch.load(checkpoint_path, map_location=device)
    here = os.path.dirname(os.path.abspath(__file__))

    if isinstance(ckpt, dict) and "model" in ckpt:
        # experiment.py saves a rich checkpoint:
        # ckpt["model"]  -> trained weights
        # ckpt["config"] -> architecture settings used during training
        # ckpt["stoi"]   -> character to token ID mapping
        # ckpt["itos"]   -> token ID to character mapping
        #
        # Use the saved config so the fresh model has the exact same shape as
        # the model that produced these weights.
        config = GPTConfig(**ckpt["config"])
        model = GPTLanguageModel(config).to(device)

        # Re-apply weight tying before loading the weights.
        # The input token embedding table maps token IDs to vectors.
        # The output head uses token vectors to score possible next tokens.
        # This line makes both places share the same learned tensor.
        model.lm_head.weight = model.token_position.token_emb.weight

        # The fresh model started with random weights. load_state_dict replaces
        # them with the trained weights from the checkpoint.
        model.load_state_dict(ckpt["model"])
        return model, ckpt["stoi"], ckpt["itos"]

    # gpt.train saves a bare state_dict: weights only, with no config or tokenizer.
    # Rebuild the tokenizer from input.txt and use the default GPTConfig.
    # This works only if the bare checkpoint was trained with the same defaults.
    text = open(os.path.join(here, "input.txt")).read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}

    # Create a fresh model with the default architecture and the vocabulary size
    # implied by input.txt, then fill it with the saved trained weights.
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
