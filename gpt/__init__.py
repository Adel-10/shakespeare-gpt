"""gpt: a small GPT-style transformer built from scratch in raw PyTorch.

Modules are added incrementally, one per learning step:
    tokenizer.py  - Step 1: character-level tokenizer  (done)
    (embeddings, attention, model, train, sample arrive in later steps)
"""

from .tokenizer import CharTokenizer

__all__ = ["CharTokenizer"]
