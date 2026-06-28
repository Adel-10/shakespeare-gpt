"""
tokenizer.py
============
A *character-level* tokenizer for our GPT-from-scratch project.

WHAT IS A TOKENIZER?
--------------------
A neural network can only do arithmetic. It multiplies matrices, adds numbers,
applies functions like softmax. It has no concept of the letter "h" or the word
"hear". So before any text can enter the model, we must turn it into numbers.

A tokenizer is the two-way bridge between *text* and *integers*:

        "hear"  --encode-->  [46, 43, 39, 56]   (model works on these)
   [46, 43, 39, 56]  --decode-->  "hear"        (we read this)

WHY *CHARACTER* LEVEL?
----------------------
There are three common granularities for the "atoms" we count:

  - character level : each character is one token. Vocab ~= 61 for our corpus.
  - word level      : each word is one token. Vocab = tens of thousands, and any
                      word not seen in training is "out of vocabulary" (OOV).
  - subword / BPE   : a middle ground (pieces like "ing", "hear"). This is what
                      real GPT models use.

We choose character level because:
  * The vocabulary is tiny (61 symbols), so the model's input/output layers are
    tiny too -> faster to train on a CPU.
  * There is NO out-of-vocabulary problem ever: every possible string is just a
    sequence of characters we already know.
  * It is the simplest possible thing, which keeps our focus on the transformer
    itself rather than on tokenization plumbing.

The trade-off: sequences are LONGER (one token per character instead of per
word), and the model has to learn spelling from scratch. For a learning project
on Shakespeare, that is totally fine.

(BPE is listed as a stretch goal at the end of the project.)
"""

from __future__ import annotations


class CharTokenizer:
    """Maps characters <-> integer ids based on a fixed vocabulary.

    The vocabulary is just "the sorted set of every unique character that
    appears in the training text". Sorting makes the mapping deterministic:
    rebuild the tokenizer from the same text and you get the same ids.
    """

    def __init__(self, text: str) -> None:
        # `set(text)` collapses the text down to its unique characters.
        # `sorted(...)` then gives us a stable, repeatable ordering.
        # Example: from "hello" -> {'h','e','l','o'} -> ['e','h','l','o'].
        self.chars: list[str] = sorted(set(text))

        # vocab_size is how many distinct symbols the model must know about.
        # It determines the size of the embedding table (Step 2) and the size
        # of the model's final output layer (Step 5).
        self.vocab_size: int = len(self.chars)

        # Two lookup tables, the heart of the tokenizer:
        #   stoi = "string to integer": char  -> id   (used when encoding)
        #   itos = "integer to string": id    -> char (used when decoding)
        # We assign id 0 to the first sorted char, 1 to the next, and so on.
        self.stoi: dict[str, int] = {ch: i for i, ch in enumerate(self.chars)}
        self.itos: dict[int, str] = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s: str) -> list[int]:
        """Text -> list of integer ids.

        We simply look up each character in `stoi`. Because the vocabulary is
        the full set of characters in the corpus, every character is guaranteed
        to be present (no OOV handling needed).

            encode("hi") -> [stoi['h'], stoi['i']]
        """
        return [self.stoi[ch] for ch in s]

    def decode(self, ids: list[int]) -> str:
        """List of integer ids -> text. The exact inverse of `encode`.

            decode([46, 47]) -> "".join([itos[46], itos[47]]) -> "hi"
        """
        return "".join(self.itos[i] for i in ids)


if __name__ == "__main__":
    # A tiny self-test you can run with:  python gpt/tokenizer.py
    # It builds a tokenizer from input.txt and proves encode/decode round-trip.
    import os

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "input.txt"), "r") as f:
        text = f.read()

    tok = CharTokenizer(text)
    print(f"vocab_size = {tok.vocab_size}")
    print(f"vocab      = {''.join(tok.chars)!r}")

    sample = "First Citizen:"
    ids = tok.encode(sample)
    print(f"encode({sample!r}) -> {ids}")
    print(f"decode(...)        -> {tok.decode(ids)!r}")

    # The crucial invariant: decode(encode(x)) must equal x for ALL of the text.
    assert tok.decode(tok.encode(text)) == text, "round-trip failed!"
    print("round-trip over the FULL corpus: OK")
