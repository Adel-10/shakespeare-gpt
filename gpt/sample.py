"""
sample.py
=========
Step 6: generate text from the trained model, one character at a time.

AUTOREGRESSIVE GENERATION
-------------------------
The model only knows how to do one thing: given some context, predict a
distribution over the NEXT character. To produce a whole passage we loop:

    1. feed the current context into the model
    2. look at the prediction for the LAST position (the next-char distribution)
    3. choose a character from it
    4. append that character to the context
    5. repeat

Each new character becomes part of the context for the next step — the model
literally writes by reading its own output. (We crop the context to the model's
block_size, since that's as far back as its positional embeddings reach.)

THE KNOBS THAT SHAPE THE OUTPUT
-------------------------------
* TEMPERATURE: we divide the logits by a number T before softmax.
    - T = 1.0  : the model's natural distribution.
    - T < 1.0  : sharpens it (the likely characters get even likelier) -> safer,
                 more repetitive text.
    - T > 1.0  : flattens it -> more surprising, more creative, more typos.
    - T -> 0   : approaches always-pick-the-top-character (greedy).

* GREEDY vs SAMPLING:
    - greedy  : always take the single most likely character (argmax).
                Deterministic, but tends to fall into dull loops ("the the the").
    - sampling: draw a character at random according to the probabilities.
                More varied and natural, at the cost of the occasional odd pick.

* TOP-K: before sampling, keep only the k most likely characters (set the rest to
    -infinity so they get zero probability). This stops the model from
    occasionally grabbing a nonsense character from the long tail, while still
    letting it choose freely among the plausible ones.
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, greedy=False):
    """Continue the sequence `idx` (shape (B, T)) by `max_new_tokens` characters.

    Returns a LongTensor of shape (B, T + max_new_tokens).
    """
    model.eval()
    block_size = model.config.block_size
    for _ in range(max_new_tokens):
        # 1. crop context to the last block_size tokens (the model can't see more)
        idx_cond = idx[:, -block_size:]
        # 2. predict; keep only the last position's logits -> (B, vocab_size)
        logits = model(idx_cond)[:, -1, :]
        # 3. temperature: T<1 sharpens, T>1 flattens. (T=1 leaves it unchanged.)
        logits = logits / temperature
        # 4. optional top-k: zero out everything except the k most likely chars
        if top_k is not None:
            kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        # 5. turn logits into probabilities and choose the next character
        probs = F.softmax(logits, dim=-1)            # (B, vocab_size)
        if greedy:
            idx_next = probs.argmax(dim=-1, keepdim=True)        # most likely
        else:
            idx_next = torch.multinomial(probs, num_samples=1)   # sample
        # 6. append and continue
        idx = torch.cat([idx, idx_next], dim=1)
    return idx

'''
generate() is the autoregressive text-generation loop.

Inputs:
- model: the trained GPT model.
- idx: a batch of token IDs with shape (B, T), where B is batch size and T is
  the current context length.
- max_new_tokens: how many new tokens to append.
- temperature: controls randomness by scaling logits before softmax.
- top_k: if set, keeps only the k most likely next-token choices.
- greedy: if True, always picks the most likely token instead of sampling.

The loop adds one token at a time:
1. 
Keep only the last block_size tokens, because the model cannot use more
context than its configured block size.
This line crops the input sequence so the model only sees the most recent `block_size` tokens:
idx_cond = idx[:, -block_size:]
Assume `idx` is a 2D tensor shaped like this: (B, T)
where: B = batch size &  T = current sequence length. 
For example, if `idx` contains one generated sequence: idx = tensor([[10, 23, 5, 8, 19, 2, 44]])
its shape is: (1, 7). 
Now break down the indexing: idx[:, -block_size:]
The first part: ':', means "take all rows", so this keeps every sequence in the batch.
The second part: -block_size, means "take columns starting from `block_size` tokens 
from the end, up to the end."

So if: block_size = 4
then: idx[:, -block_size:]
means: idx[:, -4:]
which keeps only the last 4 tokens: tensor([[8, 19, 2, 44]])

Why do this?
The model has positional embeddings for only `block_size` positions: pos_emb = Embedding(128, 128)
So if `block_size = 128`, the model can only process at most 128 positions at once.
During generation, `idx` keeps growing:
start:      10 tokens
later:      100 tokens
later:      200 tokens
later:      500 tokens
But the model cannot read all 500 tokens. It can only read the latest 128 tokens.
So this line says: Use the most recent context window only.
In plain English: idx_cond = idx[:, -block_size:] means:
For every sequence in the batch, keep only the last block_size tokens.
If `idx` is shorter than `block_size`, PyTorch just returns the whole sequence. 
So this is safe even at the beginning of generation.


2.
The cropped context is `idx_cond`, which has shape (B, T).
Run the model on that cropped context.
Here, B is the number of separate sequences in the batch and T is the current context length.
The model processes every sequence in the batch in parallel.

Take only the logits from the final time step.
The line is: logits = model(idx_cond)[:, -1, :]
This line runs the model and keeps only the prediction for the last token position.

Break it into two parts.
First: model(idx_cond)
The model receives `idx_cond`, which has shape (B, T).
The model returns logits shaped like (B, T, vocab_size).
For your model, vocab_size = 61, so the output shape is (B, T, 61).
That means: for every sequence in the batch, and for every position in the sequence,
the model predicts scores for all 61 possible next tokens.

Then this part selects from that output: [:, -1, :]
The first ':' means take all batches.
The '-1' means take only the last time position.
The last ':' means take all vocabulary logits.
So model(idx_cond)[:, -1, :] changes the shape from (B, T, vocab_size) to (B, vocab_size).

What are batches for?
A batch is multiple independent sequences being processed at the same time.
So if idx.shape == (B, T), 
then B = number of separate sequences and T = number of tokens in each sequence.

Example with B = 3:
idx = tensor([
    [10, 23, 5, 8],   # sequence 1
    [4,  9, 2, 7],    # sequence 2
    [31, 6, 1, 12],   # sequence 3
])
The model processes all 3 sequences in parallel.
model(idx_cond) returns shape (B, T, vocab_size), for example (3, 4, 61).
That means 3 sequences, 4 positions per sequence, and 61 possible next-token scores per position.
logits = model(idx_cond)[:, -1, :] keeps the last-position prediction for each sequence,
so the shape becomes (3, 61).
Now sequence 1 has 61 scores for its next character, 
sequence 2 has 61 scores for its next character,
and sequence 3 has 61 scores for its next character.
Each batch row gets one next-character distribution.

Later, idx_next = torch.multinomial(probs, num_samples=1) samples one next token per sequence.
If B = 3, idx_next has shape (3, 1), for example:
idx_next = tensor([
    [44],  # next token for sequence 1
    [12],  # next token for sequence 2
    [5],   # next token for sequence 3
])
Then idx = torch.cat([idx, idx_next], dim=1) appends each sampled token to its own sequence.

In the usual sampling case, you probably use only one prompt, 
so idx.shape == (1, T) and B = 1.
For example, in generate_text() you might give the model one prompt like: 
prompt = "ROMEO: "
The code turns that prompt into token IDs with: 
ids = [stoi[c] for c in prompt]
Then it creates the tensor with: 
idx = torch.tensor([ids], dtype=torch.long, device=device)
The extra brackets around ids are important.
If ids = [10, 22, 14, 17, 19, 3, 1], then torch.tensor([ids]) becomes:
tensor([[10, 22, 14, 17, 19, 3, 1]])
That shape is (1, T). The 1 means one prompt / one sequence in the batch.
So yes, in the normal usage of this project, there is one prompt and one prediction stream.

When we talk about multiple prompts or multiple batch rows, that means the model could handle
several prompts at once if we gave it a batch like:
idx = tensor([
    [10, 22, 14, 17, 19, 3, 1],  # "ROMEO: "
    [20, 18, 15, 13, 14, 3, 1],  # "JULIET:"
])
Then the shape would be (2, T), and the model would generate one next character for each row:
row 0 -> next character for "ROMEO: "
row 1 -> next character for "JULIET:"
But generate_text() is designed for the simple case: one prompt -> one generated text.
For this project, think of it like this:
Usually B = 1, one prompt, one 61-length next-character vector,
and one chosen next character per loop.
The batch explanation is mostly there to explain why the tensor has shape:
(B, T) instead of just (T,).
The batch dimension is still there because PyTorch models are usually 
written to handle many examples at once.

Short version: 
Batch rows are separate input sequences, each row gets its own next-character prediction.

Why only the last position?
During generation, you only need the model's 
prediction for the next token after the current context.
Example: if the context is "To be", the model may produce predictions after "T", after "To",
after "To ", after "To b", and after "To be".
But for generation, you only care about: What comes after "To be"?
That is the final position, so the code selects [:, -1, :].
In plain English, logits = model(idx_cond)[:, -1, :] means:
Run the model on the current context, then keep only the final position's next-token scores.


3.
Divide the logits by temperature before softmax.
The code is: logits = logits / temperature
Temperature is a generation setting that controls 
how "confident" or "random" the model's next-token choice is.
The model first outputs logits, which are raw scores for every possible next character.
Larger logits mean "the model thinks this character is more likely."
Temperature changes those scores before they become probabilities.

If temperature = 1.0, nothing changes because logits = logits / 1.0.
So you get the model's normal probability distribution.

If temperature < 1.0, the logits become larger in magnitude. 
For example, logits = logits / 0.5.
That makes the biggest logits much more dominant after softmax. 
The model becomes more confident and predictable.
Example effect: temperature = 0.5 means safer, cleaner, more repetitive output.

If temperature > 1.0, the logits become smaller in magnitude. 
For example, logits = logits / 2.0.
That makes the probabilities flatter. Less likely tokens get more chance to be picked.
The model becomes more random and creative, but also more error-prone.
Example effect: temperature = 1.5 means more surprising, more varied, 
and more likely to produce weird text.

A simple intuition:
low temperature  -> sharp probability distribution -> conservative output
high temperature -> flat probability distribution  -> adventurous output

Suppose the model's raw logits prefer these next characters: "e" = 5.0, "a" = 3.0, "x" = 1.0.
At normal temperature, "e" is most likely.
At low temperature, "e" becomes overwhelmingly likely.
At high temperature, "a" and even "x" get more probability than before.

Important: temperature does not change the model weights. It does not retrain the model.
It only changes how you sample from the model's output during generation.
You can call generate(model, idx, max_new_tokens=500, temperature=0.7) for more controlled text,
or generate(model, idx, max_new_tokens=500, temperature=1.3) for more random text.
One warning: temperature should be greater than 0. 
If it is 0, logits = logits / temperature breaks because division by zero is invalid.
For deterministic output, use greedy=True instead.


4.
If top_k is set, mask out all but the k highest logits so sampling 
only considers the most plausible next tokens. top_k is a sampling filter. 
It limits the model's next-token choices to only the k most likely tokens.
The code is:
if top_k is not None:
    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
    logits = logits.masked_fill(logits < kth, float("-inf"))

The model first produces logits for every possible next token.
In your case, there are 61 possible characters/tokens.
Without top_k, sampling can choose from all 61 tokens, meaning every token in the vocabulary.
With top_k = 10, sampling can only choose from the 10 most likely tokens.
The rest are set to -inf, which means after softmax they become probability 0.

Example before top_k = 3:
"a" = 8.0, "e" = 7.5, "o" = 6.0, "x" = 2.0, "q" = 0.5, "z" = -1.0
Only the top 3 are kept:
"a" = 8.0, "e" = 7.5, "o" = 6.0, "x" = -inf, "q" = -inf, "z" = -inf
Then softmax is applied with probs = F.softmax(logits, dim=-1).
After softmax, "a", "e", and "o" have probability, while "x", "q", and "z" have probability 0.
So sampling can still be random, but only among the most plausible options.

Effect of top_k:
top_k = None means sample from the whole vocabulary. 
Most flexible, but has more chance of weird low-probability tokens.
top_k = 1 means only the single best token is allowed, which is basically like greedy sampling.
top_k = 5 or top_k = 10 keeps output focused, still allows variety, and reduces random nonsense.
A very large top_k is closer to normal unrestricted sampling.

How the cutoff logit works in detail:
The line kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
finds the cutoff logit for top-k filtering.
It answers: What is the smallest logit value among the top-k logits?
That value is later used as a threshold in logits = logits.masked_fill(logits < kth, float("-inf")).
So anything below kth gets removed from sampling.

At this point, logits has shape (B, vocab_size).
Each row contains the model's scores for all possible next tokens.
Example with one batch row: logits = [[8.0, 7.5, 6.0, 2.0, 0.5, -1.0]]
logits.size(-1) returns the size of the last dimension, the vocabulary size.
In your model logits.size(-1) = 61; in the small example logits.size(-1) = 6.
min(top_k, logits.size(-1)) makes sure we do not ask for more tokens than exist.
For example, if top_k = 10 and vocab_size = 61, min(10, 61) = 10.
If someone accidentally writes top_k = 100 and vocab_size = 61, min(100, 61) = 61,
so torch.topk will not crash by being asked for more than the vocabulary size.

torch.topk(logits, min(top_k, logits.size(-1))) returns the top k largest values from each row.
Example: if logits = tensor([[8.0, 7.5, 6.0, 2.0, 0.5, -1.0]]) and top_k = 3,
then torch.topk(logits, 3) returns:
values = tensor([[8.0, 7.5, 6.0]]) and indices = tensor([[0, 1, 2]]).
The .values part keeps only the values, so we get tensor([[8.0, 7.5, 6.0]]).
Then [:, [-1]] means take all batch rows and take the last value from the top-k list,
but keep it as a 2D column.
So torch.topk(...).values[:, [-1]] returns tensor([[6.0]]).
That 6.0 is the smallest value that is still inside the top 3, so kth = tensor([[6.0]]).
Then logits = logits.masked_fill(logits < kth, float("-inf")) means:
Set every logit below 6.0 to -inf.
Original: [8.0, 7.5, 6.0, 2.0, 0.5, -1.0]
After masking: [8.0, 7.5, 6.0, -inf, -inf, -inf]

Why [:, [-1]] instead of [:, -1]?
[:, -1] would produce shape (B,), but [:, [-1]] keeps shape (B, 1).
That makes broadcasting work cleanly when comparing against logits, 
which has shape (B, vocab_size).

Short version: 
kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]] means:
For each batch row, find the lowest score among the top-k scores, 
and keep it as the cutoff threshold.
Then logits < kth finds every token whose logit is below that cutoff.
masked_fill(..., float("-inf")) sets those lower-ranked logits to negative infinity.
Later, probs = F.softmax(logits, dim=-1) turns -inf logits into probability 0.
top_k means before sampling, throw away every next-token option except the k most likely ones.
For your Shakespeare model, top_k = 10 or top_k = 20 can be useful because it keeps the text
from choosing extremely unlikely characters while still allowing some creativity.


5.
Convert logits to probabilities with softmax.
The code is: probs = F.softmax(logits, dim=-1)
Softmax turns raw logits into probabilities that add up to 1 across the vocabulary dimension.
After this step, every possible next token has a probability.
Tokens that were set to -inf by top_k become probability 0.


6.
Choose the next token, either by argmax when greedy=True or by random sampling from the probability distribution.
The code is:
if greedy:
    idx_next = probs.argmax(dim=-1, keepdim=True)
else:
    idx_next = torch.multinomial(probs, num_samples=1)

Greedy means always pick the single most likely next token.
Example: if "a" = 0.60, "b" = 0.25, and "c" = 0.15, greedy always chooses "a".
Effect of greedy=True: more deterministic, more predictable, often cleaner locally,
but it can become repetitive, get stuck in loops, and be less creative.
If you run generation twice with the same prompt and same model, 
greedy output should be the same each time.

Sampling means randomly choose according to the probability distribution.
Using the same example, "a" = 0.60, "b" = 0.25, and "c" = 0.15.
Sampling usually chooses "a", but sometimes chooses "b" or "c".
Effect of greedy=False: more varied, more natural for creative text, 
can produce surprising output, can also produce mistakes, 
and different runs can produce different text.

The main difference:
greedy   -> choose the best-looking option every time
sampling -> roll the dice using the model's probabilities

For text generation, sampling often gives better Shakespeare-like output 
because language is not always best produced by taking the single most 
likely character every time. Greedy decoding can collapse into boring or repetitive patterns.
Practical rule: use greedy=True when you want deterministic, predictable output.
Use greedy=False when you want more varied, creative output.
In this project, greedy=False with a reasonable temperature 
like 0.7 or 1.0 is usually more interesting than pure greedy generation.


7.
Append the chosen token to idx and repeat.
The code is: idx = torch.cat([idx, idx_next], dim=1)
idx_next has shape (B, 1), meaning one new token for each sequence in the batch.
idx has shape (B, T), meaning each sequence currently has T tokens.
Concatenating along dim=1 appends the new token to the sequence length dimension.
So the shape changes from (B, T) to (B, T + 1).
The loop repeats this process until it has appended max_new_tokens new tokens.

The returned tensor has shape (B, T + max_new_tokens). The @torch.no_grad()
decorator disables gradient tracking, which saves memory and compute because
generation uses the model for inference rather than training.
'''


def generate_text(model, stoi, itos, prompt="\n", max_new_tokens=500, **kwargs):
    """Convenience wrapper: take a string prompt, return a generated string."""
    device = next(model.parameters()).device
    ids = [stoi[c] for c in prompt]
    idx = torch.tensor([ids], dtype=torch.long, device=device)   # (1, len(prompt))
    out = generate(model, idx, max_new_tokens, **kwargs)[0].tolist()
    return "".join(itos[i] for i in out)
