"""
config.py
=========
Every model hyperparameter in one place, as a small dataclass.

WHY A CONFIG OBJECT?
As the model grows (embeddings -> attention -> blocks -> training), the same
handful of numbers get passed around everywhere and MUST stay consistent. If the
embedding dimension is 128 in one file and 192 in another, shapes won't line up
and nothing trains. Bundling them here means a single source of truth, and an
experiment ("what if the model were wider?") becomes a one-line change.
"""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    # --- vocabulary -------------------------------------------------------
    vocab_size: int = 61      # number of distinct tokens; set from the tokenizer.

    # --- shape / size of the model ---------------------------------------
    block_size: int = 128     # context length: how many previous characters the
                              # model may look at when predicting the next one.
                              # Also the number of positions our positional
                              # embedding table needs to cover.
    n_embd: int = 128         # embedding dimension = the "width" of the model.
                              # Every token becomes a vector of this many numbers.
    n_head: int = 4           # number of attention heads (Step 3).
    n_layer: int = 4          # number of stacked transformer blocks (Step 4).

    # --- regularization ---------------------------------------------------
    dropout: float = 0.1      # fraction of activations randomly zeroed during
                              # training, to fight overfitting (Steps 4-5).

    # NOTE: these defaults give a model of roughly ~1M parameters, which trains
    # comfortably on a CPU. We will revisit/tune them once we reach training
    # (Step 5) and can watch the train/validation loss curves.
