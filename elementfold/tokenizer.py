# ElementFold · tokenizer.py
# A tiny, dependency‑free tokenizer built on UTF‑8 bytes.
# Why bytes? They’re universal and deterministic: every Unicode string maps to
# a sequence of integers in [0,255]. Our model’s embedding uses vocab=256 by
# default, so this path is lossless and fast.
#
# API:
#   tok = SimpleTokenizer(vocab=256, max_len=1024)
#   ids = tok.encode("hello")        # → [104,101,108,108,111]
#   s   = tok.decode(ids)            # → "hello"
#
# Notes for non‑experts:
#   • If vocab==256, encode/decode is lossless UTF‑8 (within max_len).
#   • If vocab<256, bytes must be collapsed into the available range; we clamp
#     to [0, vocab−1]. Decoding then becomes best‑effort (information lost).

class SimpleTokenizer:
    def __init__(self, vocab: int = 256, max_len: int = 1024):
        self.vocab = int(vocab)           # total number of token ids the model expects (default 256)
        self.max_len = int(max_len)       # hard cap to keep inputs bounded for the model

    def encode(self, s: str) -> list[int]:
        b = (s or "").encode("utf-8")     # turn text into raw UTF‑8 bytes (deterministic and universal)
        b = b[: self.max_len]             # clip to max_len so sequences fit the model
        if self.vocab >= 256:             # common case: full byte range available → lossless
            return [int(x) for x in b]    # map each byte (0..255) to the same id
        # if vocab < 256, we must compress the space; clamp maps out‑of‑range bytes to the last id
        last = self.vocab - 1 if self.vocab > 0 else 0
        return [min(int(x), last) for x in b]

    def decode(self, ids: list[int]) -> str:
        # Map token ids back to bytes in [0,255]; modulo prevents invalid values from crashing bytes()
        raw = bytes((int(x) % 256 for x in ids))  # rebuild a byte string even if some ids were out of range
        # Best‑effort UTF‑8 decode: invalid sequences are ignored rather than raising an error
        return raw.decode("utf-8", "ignore")
