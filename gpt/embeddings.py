"""
embeddings.py
=============
Step 2: turning meaningless integer ids into meaningful vectors, and telling the
model WHERE each token sits in the sequence.

------------------------------------------------------------------------------
PART A — WHAT IS AN EMBEDDING (geometrically)?
------------------------------------------------------------------------------
After tokenization, the character 'h' is just the id 46. That integer is a bare
*label*: 46 is not "bigger" or "closer to" 45 in any meaningful way. The model
needs something it can do math with and, crucially, something it can *learn*.

An embedding table is just a big matrix of shape (vocab_size, n_embd). Row i is
the vector for token id i. "Embedding a token" means "look up its row":

        ids:        [46]                       (the character 'h')
        table[46] = [ 0.21, -1.3, 0.04, ... ]  (its n_embd-dimensional vector)

GEOMETRIC PICTURE: think of each token as a *point* (or arrow from the origin)
in an n_embd-dimensional space. At the start these points are random. During
training, gradient descent slowly slides them around so that tokens used in
similar ways drift close together, and directions between points start to encode
relationships. We can't picture 128 dimensions, but the 2-D cartoon is exactly
right: "meaning" becomes "position in space", and "similar" becomes "nearby".

ONE-HOT EQUIVALENCE (why a lookup is just a matrix multiply):
Picking row 46 is identical to multiplying a one-hot row vector by the table:

        one_hot(46) = [0, 0, ..., 1, ..., 0]   # a single 1 in slot 46
        one_hot(46) @ table  ==  table[46]

So an embedding layer is a plain linear layer whose input is a one-hot vector.
We just skip building the one-hot and index directly, because it's far cheaper.
(The notebook checks this equivalence with real numbers.)

------------------------------------------------------------------------------
PART B — WHY POSITIONAL ENCODING?
------------------------------------------------------------------------------
Here is the subtle part. The self-attention mechanism we build in Step 3 treats
its input as an unordered *set* of tokens. It compares every token to every
other token symmetrically; nothing in that computation knows that token 3 came
before token 7. Formally, attention is "permutation-equivariant": shuffle the
input tokens and you just get the same outputs in shuffled order. So to a bare
attention layer, "the cat sat" and "sat cat the" look the SAME.

But order is the whole game in language. So we must inject position information
ourselves. We do it with a *second* lookup table indexed by POSITION instead of
by token: row t is a learned vector that means "I am at position t". We add it
to the token vector:

        x[t] = token_embedding[id_at_t]  +  position_embedding[t]

WHY ADD (not concatenate)? Adding keeps the vector width at n_embd (cheaper) and
lets every dimension carry a blend of "what token" and "where". The network has
plenty of capacity to disentangle the two during training. (The original
Transformer paper used fixed sine/cosine waves for position; GPT-style models,
including this one, just *learn* the position vectors. The notebook visualizes
the sinusoidal version because it makes the "each position gets a unique
fingerprint" idea easy to see.)

SHAPES (a recurring theme — read this slowly):
  B = batch size      (how many independent sequences we process at once)
  T = time / length   (how many tokens in each sequence; T <= block_size)
  C = channels        (= n_embd, the width of each vector)
  ids in : (B, T)              -> a grid of integer ids
  embed  : (B, T, C)           -> each id replaced by its C-dim vector
A concrete size: B=4, T=8, C=32 means a (4, 8, 32) tensor = 1,024 numbers.
"""

import torch
import torch.nn as nn


class TokenAndPositionEmbedding(nn.Module):
    """Maps a batch of token ids (B, T) to vectors (B, T, C) = token + position."""

    def __init__(self, vocab_size: int, block_size: int, n_embd: int) -> None:
        super().__init__()
        # Lookup table 1: one learned vector per TOKEN.    shape (vocab_size, C)
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        # Lookup table 2: one learned vector per POSITION. shape (block_size, C)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.block_size = block_size

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: (B, T) tensor of integer ids.
        B, T = idx.shape
        assert T <= self.block_size, (
            f"sequence length {T} exceeds block_size {self.block_size}"
        )

        # Look up each token's vector:           (B, T) -> (B, T, C)
        tok = self.token_emb(idx)

        # Build the position ids [0, 1, ..., T-1] and look them up:
        #   arange(T) is (T,) -> pos_emb -> (T, C)
        pos = self.pos_emb(torch.arange(T, device=idx.device))

        # Add them. tok is (B, T, C) and pos is (T, C); PyTorch "broadcasts" pos
        # across the batch dimension, i.e. the SAME position vectors are added to
        # every sequence in the batch. Result: (B, T, C).
        return tok + pos
