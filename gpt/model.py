"""
model.py
========
Step 4 (finale): assemble the full GPT language model from the pieces we've
built. The whole model is just:

    ids ->  token + position embedding        (Step 2)
        ->  a stack of N transformer blocks    (Steps 3 & 4)
        ->  a final layer norm
        ->  a linear "language-model head" that produces a score for every
            possible next character at every position

The output is `logits` of shape (B, T, vocab_size): for each position in each
sequence, an unnormalized score for each of the 61 characters being the NEXT one.
Turning those scores into a training loss (cross-entropy) is Step 5; using them
to generate text is Step 6.
"""

import torch
import torch.nn as nn

from .config import GPTConfig
from .embeddings import TokenAndPositionEmbedding
from .block import Block


class GPTLanguageModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Step 2: ids -> vectors that carry meaning + position
        self.token_position = TokenAndPositionEmbedding(
            config.vocab_size, config.block_size, config.n_embd
        )
        '''
        BLOCKS AS HIDDEN TRANSFORMER LAYERS

        `config.n_layer` tells us how many transformer blocks to stack. You can
        think of each block as one major hidden layer/stage of the neural
        network:

            embeddings
              -> hidden Block 1
              -> hidden Block 2
              -> hidden Block 3
              -> hidden Block 4
              -> output head

        The nuance is that a transformer block is not just a simple
        `Linear -> activation` layer. Each block is a structured hidden stage
        containing several sublayers:

            LayerNorm
            Self-attention
            Residual add
            LayerNorm
            Feed-forward
            Residual add

        EACH BLOCK HAS ITS OWN PARAMETERS

        The list comprehension below creates new `Block(...)` objects. If
        `config.n_layer = 4`, then we get 4 separate block instances:

            Block 1: its own attention, feed-forward, and layer norm parameters
            Block 2: its own attention, feed-forward, and layer norm parameters
            Block 3: its own attention, feed-forward, and layer norm parameters
            Block 4: its own attention, feed-forward, and layer norm parameters

        They have the same architecture, but they do not share weights.

        Important distinction:
        - Inside one block, the same feed-forward weights are shared across all
          token positions.
        - Across different blocks, each block has its own feed-forward weights,
          attention weights, and layer norm parameters.

        So the short version is:
            same design across blocks, different learned parameters per block.
        '''
        # Steps 3-4: a stack of transformer blocks (nn.Sequential just calls them
        # in order). Each block keeps the shape (B, T, n_embd).

        # Here you are just creating a list of `config.n_layer` Block objects
        # and passing them to nn.Sequential, which will call them in order.
        # You are not calling them on x yet, simply creating the list of layers. 
        # The forward pass will call them in order.
        self.blocks = nn.Sequential(
            *[Block(config.n_embd, config.n_head, config.block_size, config.dropout)
              for _ in range(config.n_layer)]
        )
        # A final normalization before the output projection (standard in GPT-2).
        self.ln_f = nn.LayerNorm(config.n_embd)
        # The "head": project each token's vector to one score per vocabulary item.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

        # Sensible weight initialization (small random weights) helps training
        # start in a stable place. This mirrors what GPT-2 does.
        self.apply(self._init_weights)


    '''
    WEIGHT INITIALIZATION

    `self.apply(self._init_weights)` walks through every submodule inside this
    model and calls `_init_weights(module)` on it. That includes blocks,
    attention modules, feed-forward modules, linear layers, embedding layers,
    layer norms, and so on.

    This helper only changes two module types:

    1. `nn.Linear`
       Linear layers contain a weight matrix and, usually, a bias vector.

       We initialize the weight matrix with small random values:

           nn.init.normal_(module.weight, mean=0.0, std=0.02)

       This means values are sampled from a normal distribution centered at 0,
       with most values close to 0. Small random weights give the model a stable
       starting point while still letting different neurons learn different
       features.

       If the linear layer has a bias, we initialize it to zero:

           nn.init.zeros_(module.bias)

    2. `nn.Embedding`
       Embedding layers are lookup tables. Their `.weight` tensor stores one
       learned vector per token or per position, such as:

           token embedding table:    (vocab_size, n_embd)
           position embedding table: (block_size, n_embd)

       We initialize those vectors with the same small random normal values:

           nn.init.normal_(module.weight, mean=0.0, std=0.02)

    `@staticmethod` means this function does not use `self`; it only needs the
    `module` argument passed in by `self.apply(...)`.

    The trailing underscore in PyTorch functions like `normal_()` and `zeros_()`
    means "modify this tensor in place".
    '''
    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    '''
    FORWARD PASS: WHAT X LOOKS LIKE AND WHAT IT MEANS

    Symbols:
        B = batch size
        T = sequence length
        C = embedding width (`n_embd`)
        V = vocabulary size (`vocab_size`)

    In this project, C is usually 128 and V is usually 61.

    1. Input: `idx`

        Shape:
            (B, T)

        What it contains:
            Integer token IDs, such as character IDs from the tokenizer.

        What it means:
            At this point, these are just labels. For example, id 46 might mean
            the character "h", but the number 46 itself is not a useful numeric
            feature yet.

    2. After token + position embedding:

        Code:
            x = self.token_position(idx)

        Shape:
            (B, T, C)

        What it contains:
            One C-number vector for every token position.

        What it means:
            Each position now has a learned vector that combines:
            - what token/character it is
            - where it appears in the sequence

            At this stage, a vector mostly means:
                "I am this character at this position."

    3. After the stack of transformer blocks:

        Code:
            x = self.blocks(x)

        Shape:
            (B, T, C)

        What it contains:
            Context-aware hidden vectors.

        What it means:
            Each block does:
            - attention: tokens communicate with previous positions
            - feed-forward: each token vector is processed independently

            After several blocks, each x[b, t] means something closer to:
                "the model's internal understanding of this position, given the
                 token, its position, and the previous context."

            Example: in "Firs", the vector at "s" is no longer just "the token
            s"; it can also contain information gathered from "F", "i", and "r".

    4. After the final layer norm:

        Code:
            x = self.ln_f(x)

        Shape:
            (B, T, C)

        What it contains:
            The same kind of context-aware hidden vectors, but normalized.

        What it means:
            The final hidden vectors are rescaled into a stable numeric range
            before prediction. Layer norm mostly stabilizes the numbers; it does
            not change the shape.

    5. Convert hidden vectors to logits with `lm_head`:

        Code:
            logits = self.lm_head(x)

        Shape:
            (B, T, V)

        What it contains:
            One raw score for every possible next token, at every position.

        What it means:
            `self.lm_head` is the final prediction layer. It is a linear layer:

                nn.Linear(C, V)

            If C = 128 and V = 61, it maps:

                (B, T, 128) -> (B, T, 61)

            For each token position, it takes the C-dimensional hidden vector
            and computes V scores. Those scores are called logits.

            For one position:

                hidden_vector: length C
                logits:        length V

            Each logit is the model's raw score for one possible next character:

                score for token 0
                score for token 1
                ...
                score for token V-1

            Internally, the linear layer has:

                weight shape: (V, C)
                bias shape:   (V)

            For each position, it computes:

                logits = hidden_vector @ weight.T + bias

            You can read each row of `lm_head.weight` as a learned scoring rule:
                "Does this hidden vector look like a context where this token
                 should come next?"

    Whole flow:

        idx
        (B, T)
        raw token IDs

            -> token_position(idx)

        x
        (B, T, C)
        token + position vectors

            -> blocks(x)

        x
        (B, T, C)
        context-aware hidden vectors

            -> ln_f(x)

        x
        (B, T, C)
        normalized context-aware hidden vectors

            -> lm_head(x)

        logits
        (B, T, V)
        next-token scores
    '''
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: (B, T) integer ids
        x = self.token_position(idx)   # (B, T, C)
        x = self.blocks(x)             # (B, T, C)
        x = self.ln_f(x)               # (B, T, C)
        logits = self.lm_head(x)       # (B, T, vocab_size)
        return logits

    '''
    PER-TOKEN OPERATIONS AND SHARED WEIGHTS

    Both the feed-forward layers inside the blocks and the final `lm_head` are
    applied separately to every token position, but they use shared weights.

    If x has shape:

        (B, T, C)

    then a feed-forward layer inside a block behaves like:

        for every (b, t):
            x[b, t] -> same FeedForward weights -> updated x[b, t]

    Its shape stays:

        (B, T, C) -> (B, T, C)

    The final language-model head behaves like:

        for every (b, t):
            x[b, t] -> same Linear(C, vocab_size) -> logits[b, t]

    Its shape changes from hidden vectors to vocabulary scores:

        (B, T, C) -> (B, T, vocab_size)

    The difference is:

        FeedForward:
            hidden vector -> hidden vector
            C -> C
            used inside every block

        lm_head:
            hidden vector -> vocabulary scores
            C -> vocab_size
            used once at the end

    Attention is the part that mixes information across token positions. The
    feed-forward layers and `lm_head` do not mix positions; they transform each
    token vector independently.

    WHERE THE SHARED WEIGHTS ARE

    The shared weights are the parameters stored inside the module objects.

    For the final prediction layer:

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    the shared parameters are:

        self.lm_head.weight
        self.lm_head.bias

    If n_embd = 128 and vocab_size = 61, their shapes are:

        self.lm_head.weight: (61, 128)
        self.lm_head.bias:   (61,)

    `logits = self.lm_head(x)` applies that same weight matrix and bias vector
    to every token vector x[b, t].

    For the feed-forward network inside the first block, the shared parameters
    live inside the linear layers:

        self.blocks[0].ffwd.net[0].weight  # (4 * n_embd, n_embd)
        self.blocks[0].ffwd.net[0].bias    # (4 * n_embd,)
        self.blocks[0].ffwd.net[2].weight  # (n_embd, 4 * n_embd)
        self.blocks[0].ffwd.net[2].bias    # (n_embd,)

    These weights are shared across token positions inside that block.

    Important distinction:

        Shared across token positions inside one block: yes.
        Shared across different blocks: no.

    Each block has its own feed-forward, attention, and layer norm parameters.
    But within one block, the same module weights are reused for every (B, T)
    token position.
    '''
    def num_params(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())
