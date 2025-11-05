# ElementFold · tokenizer.py
# A tiny, dependency‑free tokenizer built on UTF‑8 bytes.
#
# Why bytes?
#   Every Unicode string has a canonical UTF‑8 byte sequence. Mapping each byte
#   to an integer in [0,255] is deterministic and lossless (when vocab≥256),
#   which matches the project’s default embedding size (vocab=256).
#
# Public API (unchanged):
#   tok = SimpleTokenizer(vocab=256, max_len=1024)
#   ids = tok.encode("hello")   # → [104,101,108,108,111]
#   s   = tok.decode(ids)       # → "hello"
#
# Notes for non‑experts:
#   • If vocab==256 (the default), encode/decode is lossless UTF‑8 (up to max_len).
#   • If vocab<256, bytes must be squeezed into the available range. We clamp
#     each byte into [0, vocab−1]. This makes decoding best‑effort (information
#     is lost because multiple bytes map to the same id).
#
# Small conveniences (optional; do not change defaults):
#   • encode(..., pad_to=N)   — left‑truncates to max_len, then right‑pads with 0 to length N.
#   • encode_tensor(...)      — returns a 1×L int64 tensor (imports torch lazily).
#   • decode accepts any int sequence; values are mod 256 to form valid bytes.
#
# This file intentionally has no top‑level torch import so it can be used
# in light‑weight contexts (config tools, logging, etc.).

from __future__ import annotations

from typing import Iterable, List, Optional


class SimpleTokenizer:
    def __init__(self, vocab: int = 256, max_len: int = 1024):
        """
        Args:
          vocab:  expected vocabulary size of the model’s embedding (default 256).
          max_len: maximum number of tokens produced by encode (hard cap).
        """
        self.vocab = max(1, int(vocab))     # keep within [1, ∞)
        self.max_len = max(1, int(max_len)) # avoid degenerate lengths

    # ————————————————————————————————————————————————
    # Core methods (stable contract)
    # ————————————————————————————————————————————————
    def encode(self, s: str, pad_to: Optional[int] = None) -> List[int]:
        """
        Convert a Python str → list[int] of length ≤ max_len.
        If pad_to is provided, the output is right‑padded with 0s to that length
        (after truncation to max_len). Padding uses id=0 by convention.

        Behavior when vocab < 256:
          Each UTF‑8 byte b∈[0,255] is clamped to min(b, vocab−1).
          This preserves low‑valued bytes and collapses higher ones.
        """
        b = (s or "").encode("utf-8")[: self.max_len]   # deterministic UTF‑8 bytes, truncated
        if self.vocab >= 256:
            ids = [int(x) for x in b]                   # lossless path
        else:
            ceiling = self.vocab - 1                    # last valid id
            ids = [int(x) if int(x) <= ceiling else ceiling for x in b]

        if pad_to is not None:
            # Right‑pad with zeros up to pad_to (but never exceed max_len)
            target = int(pad_to)
            if target < 0:
                target = 0
            target = min(target, self.max_len)
            if len(ids) < target:
                ids = ids + [0] * (target - len(ids))
            else:
                ids = ids[:target]  # keep deterministic length if caller requests a smaller pad_to
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        """
        Convert a sequence of ints back to a best‑effort UTF‑8 string.
        Values are mapped into [0,255] with modulo, guaranteeing valid byte construction.
        Invalid UTF‑8 sequences are ignored rather than raising an error.
        """
        raw = bytes((int(x) % 256 for x in ids))
        return raw.decode("utf-8", errors="ignore")

    # ————————————————————————————————————————————————
    # Small conveniences (optional)
    # ————————————————————————————————————————————————
    def encode_tensor(self, s: str, pad_to: Optional[int] = None):
        """
        Like encode(), but returns a 1×L int64 tensor suitable for feeding models.
        Torch is imported lazily so this module stays dependency‑free at import time.
        """
        ids = self.encode(s, pad_to=pad_to)
        # Lazy import to keep top‑level dependency‑free
        import torch  # type: ignore
        return torch.tensor([ids], dtype=torch.long)

    def decode_tensor(self, t) -> str:
        """
        Decode from a 1×L or L tensor back to text. Accepts any tensor‑like with integer values.
        """
        try:
            import torch  # type: ignore
            if isinstance(t, torch.Tensor):
                v = t.detach().cpu().view(-1).tolist()
            else:
                v = list(t)
        except Exception:
            v = list(t)
        return self.decode(v)

    # ————————————————————————————————————————————————
    # UX niceties
    # ————————————————————————————————————————————————
    def is_lossless(self) -> bool:
        """True iff vocab≥256 (byte‑perfect round‑trip for inputs ≤ max_len)."""
        return self.vocab >= 256

    def __repr__(self) -> str:
        return f"SimpleTokenizer(vocab={self.vocab}, max_len={self.max_len})"


if __name__ == "__main__":
    # Tiny self‑check (does not require torch)
    tok = SimpleTokenizer()
    msg = "hello · こんにちは · مرحبا"
    ids = tok.encode(msg)
    back = tok.decode(ids)
    print("vocab:", tok.vocab, "max_len:", tok.max_len)
    print("ids (head):", ids[:16], "… len=", len(ids))
    print("round‑trip (lossless? ", tok.is_lossless(), "):", back[:48], "…")
vesqm