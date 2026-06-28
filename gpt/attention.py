"""
attention.py
============
Step 3: self-attention — the heart of the transformer.

THE ONE-SENTENCE IDEA
---------------------
Up to now each position's vector only knows about itself. Self-attention lets
every position LOOK BACK at earlier positions and pull in whatever is relevant
to it. It is how the model builds context: when predicting the character after
"Firs", the position at "s" can look back at "F", "i", "r" and decide "this is
the word 'First'".

QUERY / KEY / VALUE  (a search-engine analogy)
----------------------------------------------
For each token we compute three different vectors, by multiplying its embedding
by three learned weight matrices:

  * QUERY (q): "what am I looking for?"          (the text you type into a search box)
  * KEY   (k): "what do I contain / offer?"      (the label on each result)
  * VALUE (v): "what do I hand over if chosen?"  (the actual content of that result)

Token i's query is compared against every token's key. Where a query and a key
are similar, that token earns a high attention weight, and we pull in more of
its VALUE. So q and k decide WHO to listen to; v is WHAT gets passed along. All
three are learned, so the model discovers for itself what "relevant" means.

WHY A DOT PRODUCT?
------------------
The similarity between a query and a key is their dot product (multiply matching
entries, then sum). A dot product is large + when two vectors point the same
way, ~0 when they're unrelated (perpendicular), negative when opposed:

   q=[1,0] . k=[1,0]  =  1   (aligned -> very relevant)
   q=[1,0] . k=[0,1]  =  0   (unrelated)
   q=[1,0] . k=[-1,0] = -1   (opposed)

Doing this for every (query, key) pair at once is a single matrix multiply:
   scores = Q @ K^T          (T, hs) @ (hs, T) = (T, T)
Entry (i, j) answers "how much should token i attend to token j?".

THE sqrt(head_size) SCALING  (subtle but important)
---------------------------------------------------
Q and K entries are ~unit-variance numbers. A dot product sums `head_size` of
them, so its variance grows to about `head_size`, i.e. its typical magnitude is
about sqrt(head_size). With head_size=64 the scores swing by ~±8 before any
learning. Feed numbers that large into softmax and it SATURATES: almost all the
weight collapses onto one token (softmax becomes nearly one-hot) and the
gradient through it goes to ~0, so the layer can barely learn. Dividing scores
by sqrt(head_size) rescales them back to ~unit size and keeps softmax soft and
trainable. (The notebook demonstrates this with real numbers.)

THE CAUSAL MASK  (no peeking at the future)
-------------------------------------------
We predict the NEXT character. During training we feed whole sequences and ask
the model to predict every next-character simultaneously (efficient) — but that
is only honest if position i cannot see positions after i. So before softmax we
set every "future" score (j > i) to -infinity; exp(-inf) = 0, so those get
exactly zero weight. The attention matrix becomes lower-triangular: token i
attends only to tokens 0..i.

MULTI-HEAD  (several conversations at once)
-------------------------------------------
One head learns one notion of relevance. Language has many at once (subject-verb
agreement, matching quotes/brackets, recent vs. distant context...). So we run
several SMALLER heads in parallel — each with its own Q/K/V, hence its own idea
of "relevant" — then concatenate their outputs and mix them with one more linear
layer. With n_embd=128 and n_head=4, each head works in a 32-dim subspace, so
multi-head costs about the same as one big head but captures several relationship
types simultaneously.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """A single self-attention head."""

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        # Three learned linear maps (bias-free, as is conventional here). Each
        # turns a (..., n_embd) vector into a (..., head_size) vector.
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # The causal mask: a (block_size, block_size) lower-triangular matrix of
        # 1s. Registered as a BUFFER — state that travels with the model (to
        # GPU/CPU, into saved checkpoints) but is NOT trained.
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size
        self.last_attn = None  # most recent attention weights, kept for plotting

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        '''
        What q, k, and v actually are:

        The code does not explicitly know what a token "needs" or what a token
        "represents". These three Linear layers only define a learnable
        mechanism. At initialization, q/k/v are just random learned projections
        of the token vectors.

        The analogy is:

          q = a learned "what am I looking for?" representation
          k = a learned "what do I offer?" representation
          v = a learned "what information should I pass along?" representation

        But those meanings are not hand-written into the code. They emerge from
        training. The model predicts the next character, compares that prediction
        to the true next character, and backpropagation adjusts the query, key,
        and value matrices. If attending to a previous token improves the
        prediction, the weights that produced that useful attention pattern are
        reinforced. If an attention pattern is useless, training nudges those
        weights away.

        So q/k decide which tokens align with each other, and v contains the
        information that will be gathered once the attention weights are known.
        '''

        # Relevance scores between every pair of positions, scaled by sqrt(hs):
        #   (B, T, hs) @ (B, hs, T) -> (B, T, T)
        scores = q @ k.transpose(-2, -1) * self.head_size ** -0.5

        # Causal mask: forbid attending to the future by setting those scores
        # to -inf (so softmax assigns them zero weight).
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        # Softmax turns each row into weights that are >= 0 and sum to 1.
        attn = F.softmax(scores, dim=-1)       # (B, T, T)
        self.last_attn = attn.detach()         # stash for inspection/visualization
        attn = self.dropout(attn)

        '''
        Where tokens actually use previous tokens:

        The attention weights in attn are a (B, T, T) tensor. For each sequence
        in the batch, each row says: for this query token, how much should I use
        each key token?

        The causal mask above has already forced future positions to have zero
        weight after softmax. So row i can only put nonzero weight on tokens
        0..i.

        Example for token 5 after masking and softmax:

          attn row 5 = [0.05, 0.10, 0.00, 0.60, 0.15, 0.10, 0, 0, ...]

        Then the next line computes:

          new token 5 vector =
              0.05 * value(token 0)
            + 0.10 * value(token 1)
            + 0.00 * value(token 2)
            + 0.60 * value(token 3)
            + 0.15 * value(token 4)
            + 0.10 * value(token 5)

        That is the exact point where each token pulls information from previous
        tokens. The result is a new context-aware vector for every position.
        '''
        # Weighted sum of values: each position becomes a blend of the values it
        # attended to.   (B, T, T) @ (B, T, hs) -> (B, T, hs)
        out = attn @ v
        return out


class MultiHeadAttention(nn.Module):
    """Several attention heads in parallel, concatenated and projected."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)]
        )
        # After concatenating the heads, each token vector is back to width
        # n_embd. For example, with n_embd=32 and n_head=4, each head returns
        # 8 numbers and concatenation gives:
        #
        #   [head0_result | head1_result | head2_result | head3_result]
        #
        # At that point the heads are just placed side-by-side. This projection
        # is a learned n_embd -> n_embd mixing layer that lets information from
        # different heads combine while keeping the output width unchanged.
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Run every head and concatenate along the channel dimension:
        #   n_head * (B, T, head_size)  ->  (B, T, n_embd)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # Mix the side-by-side head outputs into one normal n_embd-wide token
        # vector, then apply dropout during training for regularization.
        out = self.dropout(self.proj(out))
        return out
