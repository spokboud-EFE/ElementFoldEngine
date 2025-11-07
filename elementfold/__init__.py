# elementfold/__init__.py
# Public import surface for the ElementFold coherence engine.
# We expose just the essentials so non‑experts have a clean, stable API.

__meta__    = "⟲ ElementFold ⟲ δ⋆=0.030908106561043047"  # identity string with the coherence click
__version__ = "0"                                     # semantic version of this package snapshot

from .core.runtime import Engine       # orchestrator: fit / infer / steer / save / load
from .utils.config  import Config       # typed configuration carrier (JSON)
from .core.train   import train_loop   # end‑to‑end training loop (functional entry)
from .core.infer   import infer_loop   # inference utility (greedy / sampling decode)

__all__ = [                       # what `from elementfold import *` yields
    "Engine",
    "Config",
    "train_loop",
    "infer_loop",
]
