"""
train.py
========
Step 5: teach the model. We turn the model's predictions into a single number
(the cross-entropy LOSS) that says how wrong it is, then repeatedly nudge every
parameter a little in the direction that lowers that number (gradient descent,
driven by the Adam optimizer).

THE LOSS: CROSS-ENTROPY  (a "surprise" meter)
---------------------------------------------
At each position the model outputs `logits` = a raw score for every possible next
character. Softmax turns those into probabilities. Cross-entropy loss is simply:

    loss = average over all positions of  -log(probability assigned to the TRUE next char)

Read -log(p) as "surprise": if the model gave the correct character p=0.9, the
surprise is -log(0.9)=0.11 (small, good). If it gave it p=0.01, surprise is
-log(0.01)=4.6 (large, bad). A brand-new model is just guessing uniformly among
the 61 characters, so its loss should start near -log(1/61)=log(61)=4.11 — a handy
sanity check.

BACKPROP + GRADIENT DESCENT  (which way is downhill?)
----------------------------------------------------
The loss is one number that depends on all ~0.8M parameters. `loss.backward()`
uses the chain rule to compute, for every parameter, its GRADIENT: "if I increase
this parameter slightly, does the loss go up or down, and how fast?" We then step
every parameter a little in the downhill direction (`optimizer.step()`). Repeat
thousands of times and the loss slides down. (PyTorch's autograd computes all the
gradients for us; we never do calculus by hand.)

ADAM / ADAMW
------------
Plain gradient descent uses one fixed step size for every parameter. Adam keeps a
short running memory of each parameter's recent gradients and adapts a per-
parameter step size (momentum + scaling), which makes training faster and far
less fussy about the learning rate. AdamW is Adam with proper weight decay
(a gentle pull of weights toward zero — light regularization).

OVERFITTING  (the thing to watch on our tiny corpus)
----------------------------------------------------
We hold out the last 10% of the text as a VALIDATION set the model never trains
on. If the training loss keeps dropping while the validation loss flattens or
rises, the model is memorizing the training text rather than learning Shakespeare
in general — overfitting. With only ~98k characters and ~0.8M parameters we
expect this. Guards already in place: dropout, a small model, gradient clipping;
plus we can keep the checkpoint at the BEST validation loss (early stopping).
"""

import torch
import torch.nn.functional as F

from .config import GPTConfig
from .model import GPTLanguageModel


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"         # NVIDIA GPU — fastest option if available
    if torch.backends.mps.is_available():
        return "mps"          # Apple Silicon GPU — much faster than CPU
    return "cpu"              # CPU — slowest option, but works everywhere

'''
device chooses the hardware used for training.
model.to(device) in: model = GPTLanguageModel(config).to(device), moves the model there.
x.to(device), y.to(device) moves the data there.
'''

def get_batch(data, block_size, batch_size, device):
    """Sample `batch_size` random chunks. x is the input, y is x shifted by one
    (so y[t] is the character that should follow x[t])."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])           # (B, T)
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])   # (B, T)
    return x.to(device), y.to(device)


'''
get_batch samples random training examples from a long 1D tensor of token IDs.

It does this: ix = torch.randint(len(data) - block_size, (batch_size,))
This picks batch_size random starting positions from data.

Then for each start position i, it creates:
x = data[i:i + block_size]
y = data[i + 1:i + 1 + block_size]

So x is a chunk of tokens, and y is the same chunk shifted one token forward.
The model sees x and learns to predict y.

Example:
data = [10, 20, 30, 40, 50]
block_size = 3
x = [10, 20, 30]
y = [20, 30, 40]

y is needed because the model needs the correct answer for each position in x.
For every token in x, y contains the next token the model should have predicted.
The loss function compares the model's predictions against y, measures how wrong
the predictions were, and uses that error to update the model during training.

It returns: return x.to(device), y.to(device)

So the return value is a tuple: (x, y)
where both are PyTorch tensors on the requested device, shaped:
(batch_size, block_size)
In this code's comments, that is (B, T): B is batch size, T is sequence length.
'''


@torch.no_grad()
def estimate_loss(model, splits, block_size, batch_size, eval_iters, device):
    """Average the loss over a few batches of each split. Uses model.eval() so
    dropout is OFF during measurement (we want a clean, deterministic estimate)."""
    out = {}
    model.eval()
    for name, data in splits.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            logits = model(x)
            B, T, V = logits.shape
            losses[k] = F.cross_entropy(logits.view(B * T, V), y.view(B * T)).item()
        out[name] = losses.mean().item()
    model.train()
    return out


'''
estimate_loss measures how well the model is doing on both the training data
and validation data, without actually training the model.

The important idea is:
Run the model on a few random batches, calculate the loss for each batch,
average those losses, and return the result.
It tells you approximately how wrong the model currently is.

What It Returns: It returns a dictionary like this:
{
    "train": 2.13,
    "val": 2.25
}
The exact numbers will change, but usually:
out[name] = losses.mean().item()
means each split gets one average loss value.
So the return value is: out
where out maps each split name, like "train" or "val", to the model's average
loss on that split.

How It Works

First:
out = {}
model.eval()

out stores the final results.

model.eval() switches the model into evaluation mode. This matters because some
layers behave differently during training vs evaluation. In your code, dropout
is the main example. During training, dropout randomly disables parts of the
network. During evaluation, dropout is turned off so the loss measurement is
cleaner and more stable.

Then: 
for name, data in splits.items():
splits probably contains something like:
{
    "train": train_data,
    "val": val_data
}
So the function loops over both datasets.

For each one:
losses = torch.zeros(eval_iters)
This creates a tensor to store multiple loss values. If eval_iters = 200, it
will measure 200 random batches and store 200 losses. So losses is a 1D tensor
with shape: (eval_iters,)
For example, if eval_iters = 200, losses.shape == (200,)
Each slot stores one scalar loss for one randomly sampled evaluation batch:
losses[0] = average loss over random batch 0
losses[1] = average loss over random batch 1
losses[2] = average loss over random batch 2
...

Then:
for k in range(eval_iters):
    x, y = get_batch(data, block_size, batch_size, device)

This samples one random batch from the current split.
x is the input tokens.
y is the correct next-token targets.
Important detail: get_batch samples random starting positions using torch.randint.
That means it samples with replacement.

So yes:
- the same starting position can appear more than once
- the same batch could theoretically be sampled again
- some possible chunks may not be sampled during evaluation
- some chunks may be sampled multiple times
That is normal here. estimate_loss is not trying to calculate the exact loss
over every possible chunk in the dataset. It is trying to estimate the loss
cheaply using random samples.
That is why it is called estimate_loss, not calculate_exact_loss.
The tradeoff is:
more eval_iters = more accurate estimate, slower evaluation
fewer eval_iters = noisier estimate, faster evaluation

Then:
logits = model(x)
The model makes predictions for every token position in x.
The output shape is:
B, T, V = logits.shape, Where:
B = batch size
T = block size / sequence length
V = vocab size

So if: batch_size = 32, block_size = 64, vocab_size = 65
then: logits.shape == (32, 64, 65)
That means for every token position in every example, the model outputs a score
for every possible vocabulary token.
You can think of logits like this:
[
  [T vectors, each length V],  # batch example 1
  [T vectors, each length V],  # batch example 2
  ...
  [T vectors, each length V],  # batch example B
]
Since V = 65 in this example, each token position has a vector of 65 scores.
So conceptually: B groups, each group has T vectors, each vector has length 65
The target y has shape: (B, T)
For example: y.shape == (32, 64)
Each value in y is the correct next-token ID for that position.

Then:
losses[k] = F.cross_entropy(logits.view(B * T, V), y.view(B * T)).item()
This calculates how wrong the model was.

The line does three things:
1. reshapes the model predictions
2. reshapes the correct answers
3. calculates one loss number and stores it in losses[k]

cross_entropy expects its first argument to be predictions shaped like: (N, C)
where:
N = number of predictions
C = number of classes

For this model: N = total token positions in the batch, C = vocab size
So we want: (B * T, V)
Example: (32 * 64, 65), which is: (2048, 65)
That means we are treating the batch as 2048 separate next-token prediction
problems.
This part: logits.view(B * T, V), changes logits from: (B, T, V) -> (B * T, V)
Example: (32, 64, 65) -> (2048, 65)
Before reshaping, logits is like:
[
  [T vectors of length 65],
  [T vectors of length 65],
  ...
  [B times]
]

After reshaping, it is compressed into one list:
[
  vector length 65,
  vector length 65,
  vector length 65,
  ...
]
There are B * T vectors total.
So:
before: [B batches] x [T positions] x [65 scores]
after:  [B * T positions] x [65 scores]

The second argument to cross_entropy should be the correct class labels shaped
like: (N). So y needs to become: (B * T). Example: (2048)
This part: y.view(B * T), changes y from: (B, T) -> (B * T)
Example: (32, 64) -> (2048)
Each value is an integer token ID, like: [12, 4, 33, 8, ...]

So now cross_entropy receives:
F.cross_entropy(
    logits.view(B * T, V),  # predictions: (2048, 65)
    y.view(B * T)           # correct answers: (2048)
)
This turns the whole batch into one long list of next-token prediction problems.
Each row in logits.view(B * T, V) lines up with one target in y.view(B * T):
prediction vector 0 -> correct token 0
prediction vector 1 -> correct token 1
prediction vector 2 -> correct token 2
...
For each of the B * T token positions, cross_entropy compares:
the model's V scores (from logits) against the correct token ID (from y).
For example, one prediction row might look like: [0.2, -1.4, 3.1, 0.7, ...]
And the correct answer might be: 2
That means the correct next token is token ID 2.
cross_entropy rewards the model if score 2 is high, and penalizes it if score 2
is low.

By default, F.cross_entropy returns the average loss across all B * T
predictions. So if batch_size = 32 and block_size = 64, one losses[k] value is
already the average loss over: 32 * 64 = 2048 next-token predictions.
So: losses[k] stores one scalar loss for the k-th evaluation batch.

losses[0] = average loss over the first random group of examples
losses[1] = average loss over the second random group of examples
losses[2] = average loss over the third random group of examples
...

The .item() part turns the scalar PyTorch tensor, like tensor(2.4837), into a
normal Python number, like 2.4837, before storing it in losses[k].
After all batches are measured: out[name] = losses.mean().item()
This averages all the batch losses for that split.

Finally:
model.train()
return out

model.train() switches the model back into training mode so training can
continue normally.

Why @torch.no_grad() Is Used? 
@torch.no_grad() means PyTorch should not track gradients while this function runs.
That matters because estimate_loss is only measuring performance. It is not
updating the model. Without torch.no_grad(), PyTorch would waste memory and 
computation tracking operations for backpropagation, even though 
you never call .backward() inside this function. So this makes evaluation faster and lighter.

Why We Need estimate_loss? 

During training, the model's loss on one random batch can be noisy. One batch
might be easy, another might be hard. estimate_loss gives a more stable measurement 
by averaging over many batches. It also checks both: training loss & validation loss. 
Training loss tells you how well the model is fitting the data it trains on.
Validation loss tells you how well the model performs on data it is not directly
training on. That helps you detect overfitting. 
For example:
train loss keeps going down
val loss starts going up
That usually means the model is memorizing the training data instead of learning
patterns that generalize.
'''


def train(text, config=None, max_iters=5000, eval_interval=500, eval_iters=200,
          batch_size=32, learning_rate=1e-3, device=None, seed=1337, verbose=True):
    """Train a GPT on `text`. Returns (model, history, meta) where history is a
    list of (iter, train_loss, val_loss) and meta = (stoi, chars)."""
    torch.manual_seed(seed)
    device = device or get_device()

    # --- tokenizer + data ---
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    data = torch.tensor(encode(text), dtype=torch.long)

    # --- 90/10 train/validation split ---
    n = int(0.9 * len(data))
    splits = {"train": data[:n], "val": data[n:]}

    config = config or GPTConfig(vocab_size=vocab_size)
    model = GPTLanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if verbose:
        print(f"device={device}  params={model.num_params():,}  iters={max_iters}")

    history = []
    for it in range(max_iters + 1):
        # periodically measure train & val loss
        if it % eval_interval == 0 or it == max_iters:
            losses = estimate_loss(model, splits, config.block_size, batch_size, eval_iters, device)
            history.append((it, losses["train"], losses["val"]))
            if verbose:
                print(f"iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}")

        # --- one optimization step (the heart of training) ---
        x, y = get_batch(splits["train"], config.block_size, batch_size, device)
        logits = model(x)                                  # forward pass
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), y.view(B * T))

        optimizer.zero_grad(set_to_none=True)              # clear last step's gradients
        loss.backward()                                    # backprop: fill in new gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # guard vs exploding grads
        optimizer.step()                                   # nudge every parameter downhill

    return model, history, (stoi, chars)


'''
train is the main training function for the GPT model.
It takes raw text, turns that text into token IDs, creates the model, trains the
model for many iterations, occasionally measures train/validation loss, and then
returns the trained model plus some useful training information.

The main inputs are:
text: the raw training text, like the Shakespeare text.
config: the GPTConfig object that controls model size, block_size, vocab_size,
        number of layers, embedding size, etc. If config is None, the function
        creates a default config.
max_iters: how many training steps to run.
eval_interval: how often to call estimate_loss and print progress.
eval_iters: how many random groups of batches estimate_loss should average over.
batch_size: how many random batches get trained on per optimization step.
learning_rate: how large each optimizer update should be.
device: "cpu", "cuda", or "mps". If None, get_device() picks one automatically.
seed: makes the random choices repeatable.
verbose: controls whether training progress gets printed.

The function returns:
return model, history, (stoi, chars)
model is the trained GPTLanguageModel.
history is a list of loss measurements collected during training, each item is: 
(iteration_number, train_loss, val_loss). For example:
[
    (0, 4.23, 4.25),
    (500, 2.41, 2.46),
    (1000, 2.12, 2.20),
]
stoi means "string to integer". It is a dictionary that maps each character to
its token ID. chars is the sorted list of unique characters in the training text.
Together, (stoi, chars) are useful later for encoding text before generation and
decoding model outputs back into characters.

How It Works:
First:
torch.manual_seed(seed)
device = device or get_device()

torch.manual_seed(seed) makes PyTorch's random behavior repeatable. This affects
things like random batch sampling and model initialization.
device = device or get_device() means: if the caller passed a device, use it
otherwise, automatically choose the best available device
get_device() prefers CUDA, then Apple Silicon MPS, then CPU.

Tokenizer And Data:
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
data = torch.tensor(encode(text), dtype=torch.long)

chars gets every unique character in the text and sorts them.
For example, if the text only contained: "abca"
then: chars = ["a", "b", "c"], vocab_size = 3
stoi maps each character to a number:
{
    "a": 0,
    "b": 1,
    "c": 2
}
encode turns text into token IDs.
For example: encode("abca") -> [0, 1, 2, 0]
Then data becomes a PyTorch tensor of token IDs: tensor([0, 1, 2, 0])
The dtype is torch.long because embedding layers and cross_entropy expect token
IDs / class labels to be integer tensors.

Train/Validation Split: 
n = int(0.9 * len(data))
splits = {"train": data[:n], "val": data[n:]}

This splits the tokenized data into: 90% training data & 10% validation data
The model updates its weights using only the training split.
The validation split is kept separate so estimate_loss can check how well the
model performs on text it is not directly training on.

Create The Model And Optimizer:
config = config or GPTConfig(vocab_size=vocab_size)
model = GPTLanguageModel(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

If no config was passed in, the function creates a default GPTConfig and sets
the vocab_size to match the training text.
GPTLanguageModel(config) creates the neural network.
.to(device) moves the model weights to the CPU, GPU, or MPS device.
An optimizer is the part of training that updates the model's weights.
The rough flow is:
logits = model(x)
loss = F.cross_entropy(...)
loss.backward()
optimizer.step()

loss.backward() calculates gradients. A gradient says: if you change this
parameter in this direction, the loss should go down. The optimizer uses those
gradients to actually change the parameters. So backward() calculates how to
change the weights, and optimizer.step() changes the weights.

Adam is an optimizer algorithm. Basic gradient descent updates parameters like:
parameter = parameter - learning_rate * gradient
Adam is smarter than basic gradient descent because it keeps track of two moving
averages for each parameter:
1. average gradient direction
2. average squared gradient size
This helps Adam adjust the update size for each parameter individually. So
instead of every parameter getting the exact same style of update, Adam adapts
based on the recent gradient behavior of each parameter. That usually makes
training faster and more stable than plain gradient descent.

AdamW is a variant of Adam. The W stands for weight decay. Weight decay is a
regularization technique that gently discourages weights from becoming too
large. Large weights can make the model overfit or behave unstably. AdamW
applies weight decay in a cleaner way than the original Adam implementation.
In modern deep learning, AdamW is commonly preferred over Adam, especially for
transformer models like GPT.

So: Adam = adaptive optimizer, AdamW = Adam with better weight decay behavior.
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
This creates an AdamW optimizer that will update your model's parameters.
torch.optim.AdamW(...) means: use PyTorch's AdamW optimizer.
model.parameters() gives the optimizer all trainable parameters in the model.
That includes things like token embeddings, position embeddings, attention
weights, feedforward weights, layer norm weights, and final output projection
weights. Basically, all tensors inside the model that have requires_grad=True.
lr=learning_rate sets the learning rate, which controls how big each update
step is. If it is too small, training is slow. If it is too large, training can
become unstable or fail to learn.
So this line means: create an AdamW optimizer, give it all trainable model
weights, and tell it to update them using the chosen learning rate.

Training History:
history = []

This list stores loss measurements over time.
During training, every eval_interval steps, the code appends:
(it, losses["train"], losses["val"])
This gives you a record of how training loss and validation loss changed as the
model trained.

Main Training Loop:
for it in range(max_iters + 1):
This runs the training loop from iteration 0 through max_iters.
The +1 means if max_iters = 5000, the loop includes iteration 5000.

Periodic Loss Measurement: Inside the loop:
if it % eval_interval == 0 or it == max_iters:
    losses = estimate_loss(...)
    history.append((it, losses["train"], losses["val"]))
This periodically pauses normal training to estimate the current train and
validation loss. it % eval_interval == 0 means:
run evaluation every eval_interval iterations. 
or it == max_iters means: always evaluate on the final iteration.
estimate_loss does not train the model. It only measures how wrong the model is
on random batches from the train and validation splits.

One Optimization Step:
After optional evaluation, the function does one training step:
x, y = get_batch(splits["train"], config.block_size, batch_size, device)
This samples a random batch from the training split.
x is the model input, shaped: (batch_size, block_size)
y is the correct next-token target, also shaped: (batch_size, block_size)
Then: logits = model(x)
This is the forward pass. The model looks at x and produces predictions.
logits has shape: (B, T, V), where:
B = batch size
T = block size / sequence length
V = vocab size

Then: 
B, T, V = logits.shape
loss = F.cross_entropy(logits.view(B * T, V), y.view(B * T))

This calculates the training loss for the current batch.
logits.view(B * T, V) reshapes the predictions from: (B, T, V) -> (B * T, V)
y.view(B * T) reshapes the targets from: (B, T) -> (B * T)
This is needed because cross_entropy expects:
predictions: (number_of_predictions, number_of_classes)
targets:     (number_of_predictions)
For a character GPT, the number of classes is the vocabulary size.
So each token position becomes one next-token prediction problem.

Backpropagation And Parameter Update:
optimizer.zero_grad(set_to_none=True)
This clears the gradients from the previous training step.
PyTorch accumulates gradients by default, so if we did not clear them, the new
gradients would be added on top of the old gradients.

Then:
loss.backward()

This is backpropagation.
It calculates gradients for the model parameters. A gradient tells each
parameter which direction would reduce the loss.

Then:
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
This clips the gradient norm so the update does not become too large.
It is a guard against exploding gradients, where gradients become huge and make
training unstable.

Then:
optimizer.step()
This applies the update. AdamW looks at the gradients and nudges every model
parameter in a direction that should reduce the loss.

In short, each training iteration does this:
1. maybe estimate train/validation loss
2. sample a random training batch
3. run the model forward
4. calculate cross_entropy loss
5. clear old gradients
6. run backpropagation
7. clip gradients
8. update model parameters

Why We Need train
The train function ties the whole training process together.
get_batch only creates examples.
estimate_loss only measures performance.
The model only defines the neural network.
train coordinates everything:
it prepares the data
it creates the model
it creates the optimizer
it repeatedly samples batches
it calculates loss
it updates weights
it tracks progress with train and validation loss
it returns the trained model and tokenizer information
'''


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    text = open(os.path.join(here, "input.txt")).read()
    model, history, meta = train(text)
    torch.save(model.state_dict(), os.path.join(here, "shakespeare_gpt.pt"))
    print("saved checkpoint -> shakespeare_gpt.pt")
