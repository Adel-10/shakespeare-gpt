"""
block.py
========
Step 4: the three pieces that turn raw attention into a real, stackable
transformer block — a feed-forward layer, residual connections, and layer norm.

THE RHYTHM: COMMUNICATE, THEN COMPUTE
-------------------------------------
Attention (Step 3) lets tokens *communicate* — each position gathers information
from earlier positions. But attention is mostly a weighted average; it doesn't do
much per-token "thinking". So after each attention step we add a FEED-FORWARD
layer that processes every token independently. The block alternates:
    1. communicate:  multi-head self-attention
    2. compute:      a per-token feed-forward network
We'll wrap each of those two sub-steps in a residual connection and feed each a
layer-normalized input.

------------------------------------------------------------------------------
FEED-FORWARD  (per-token computation)
------------------------------------------------------------------------------
Two linear layers with a non-linearity between them. It expands the width from
n_embd to 4*n_embd, applies the non-linearity, then projects back down:
    (B,T,n_embd) -> (B,T,4*n_embd) -> ReLU -> (B,T,n_embd)
The 4x expansion gives the network room to compute richer intermediate features
(it's where most of the model's parameters live). This runs on each token
separately and identically — no mixing between positions; that already happened
in attention.

WHY THE NON-LINEARITY? Without it (two linear layers back-to-back), the whole
thing collapses to a SINGLE linear map — stacking linear layers buys you nothing.
The ReLU (keep positives, zero out negatives) is what lets the model represent
non-linear functions. (GPT-2 uses GELU, a smooth version of ReLU; ReLU is the
simplest thing that works and is easiest to reason about.)

------------------------------------------------------------------------------
RESIDUAL CONNECTIONS  (the gradient highway)
------------------------------------------------------------------------------
Instead of `x = sublayer(x)` we write `x = x + sublayer(x)`. That little `+ x`
is the residual (a.k.a. skip connection).

WHY: deep stacks are hard to train because gradients get multiplied layer after
layer on the way back and tend to vanish (-> 0) or explode (-> huge). The `+ x`
gives the gradient a direct path straight through every layer, so even deep
models train. It also reframes each sublayer's job: rather than producing the
whole next representation from scratch, it only has to learn a small *adjustment*
to add to what came in, which is a much easier thing to optimize.

WHAT BREAKS WITHOUT IT: a deep network barely trains at all — the loss gets stuck
because the early layers receive almost no useful gradient.

------------------------------------------------------------------------------
LAYER NORM  (keep activations well-scaled)
------------------------------------------------------------------------------
Layer norm takes each token's vector and rescales it to mean 0 and standard
deviation 1 *across its features*, then applies a learned scale and shift (so the
model can undo the normalization if it wants to). It's the z-score idea: like
turning raw exam marks into "how many standard deviations above average", so the
numbers are comparable regardless of the test's difficulty.

WHY: as activations flow through many layers their scale tends to drift —
ballooning or shrinking — which makes training unstable and slow. Re-normalizing
at each step keeps every layer's input in a sane, consistent range, so training
is stable and far less fussy about initialization and learning rate.

WHAT BREAKS WITHOUT IT: activation magnitudes drift across depth (the notebook
plots this), gradients become unreliable, and training is unstable or stalls.

PRE-NORM: we normalize the INPUT to each sublayer (`x + sublayer(norm(x))`) rather
than normalizing the output. This "pre-norm" arrangement is what modern GPTs use;
it keeps the residual path clean and is markedly more stable to train.

(Aside: layer norm normalizes each token on its own, across features. That's
different from batch norm, which normalizes across the batch — layer norm is the
right choice for sequences, where batches are small and lengths vary.)
"""

import torch.nn as nn

from .attention import MultiHeadAttention


class FeedForward(nn.Module):
    """Per-token computation: expand to 4x width, non-linearity, project back."""

    def __init__(self, n_embd: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),   # (B,T,C) -> (B,T,4C)
            nn.ReLU(),                       # the crucial non-linearity
            nn.Linear(4 * n_embd, n_embd),   # (B,T,4C) -> (B,T,C)
            nn.Dropout(dropout),             # regularization (fights overfitting)
        )

    def forward(self, x):
        return self.net(x)

'''
    `FeedForward(n_embd)` only needs the vector width C (= n_embd). It does not
    need B or T because `nn.Linear` applies to the last dimension and carries any
    leading dimensions along unchanged:
    (B, T, C) -> Linear(C, 4C) -> (B, T, 4C)
    (B, T, 4C) -> Linear(4C, C) -> (B, T, C)
    So one shared feed-forward module processes every token vector in the (B,T)
    grid, regardless of batch size or sequence length.

    B, T, C = 4, 16, 128
    x = torch.randn(B, T, C)
    ff = FeedForward(C)
    out = ff(x)

    What happens:
    x:                (4, 16, 128)
    Linear(128,512):  (4, 16, 512)
    ReLU:             (4, 16, 512)
    Linear(512,128):  (4, 16, 128)
    out:              (4, 16, 128)

    So the key idea is:
    FeedForward(C) defines how to process one C-wide token vector.
    ff(x) applies that same processing to every token vector in the (B,T) grid.

    B and T are not parameters of the layer 
    because the same layer works for any batch size and any sequence length.
'''


class Block(nn.Module):
    """One transformer block: communicate (attention) then compute (feed-forward),
    each wrapped in a pre-norm residual connection."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))     # 1) communicate between tokens
        x = x + self.ffwd(self.ln2(x))   # 2) compute on each token
        return x

'''
    RESIDUAL CONNECTIONS IN THIS BLOCK

    A residual connection means:
        new_x = old_x + learned_update

    Instead of replacing x completely, each sublayer only learns an update
    to add to the representation that already exists. This preserves useful
    information and gives gradients a direct path backward through deep
    stacks of blocks.

    Why we need this:
    - During training, gradients flow backward from the loss through every
        block. Without residual paths, those gradients can vanish toward zero
        or explode to huge values after many layers.
    - With `x + update`, even if the update is initially poor, the original
        x can still pass through. A block can also learn to change very little
        by producing an update close to zero.
    - This makes optimization easier: each sublayer learns "what to adjust",
        not "how to rebuild the whole representation from scratch".

    Line 1:
        x = x + self.sa(self.ln1(x))

    What is computed:
    - `self.ln1(x)` normalizes each token vector across its C features. The
        shape stays (B, T, C).
    - `self.sa(...)` runs multi-head self-attention on the normalized x. This
        is the "communicate" step: each token position gathers information
        from previous token positions. The output shape is still (B, T, C).
    - `x + ...` adds that communication update back to the original x. The
        old representation is preserved, and attention contributes a learned
        context update.

    Line 2:
        x = x + self.ffwd(self.ln2(x))

    What is computed:
    - Now x already includes the attention update from line 1.
    - `self.ln2(x)` normalizes this updated representation, again keeping
        shape (B, T, C).
    - `self.ffwd(...)` runs the feed-forward network independently on each
        token vector. This is the "compute" step: every token gets its own
        per-position update, using shared weights. The output shape is (B,T,C).
    - `x + ...` adds that computation update back to the current x.

    The additions work because both sublayers return the same shape as x:
        x:              (B, T, C)
        attention out:  (B, T, C)
        feedforward out:(B, T, C)

    So this block refines x in two stages:
        old x + communication update + computation update
'''