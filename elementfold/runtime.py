# ElementFold · runtime.py
# ──────────────────────────────────────────────────────────────────────────────
# The Engine is a small orchestration spine. It gives you one place to:
#   • fit()               — train a model via the project’s default training loop,
#   • infer(...)          — run inference (auto‑materialize model if needed),
#   • steer(...)          — route human intent through a steering controller + adapter,
#   • apply_control(...)  — push (β, γ, ⛔) down into model blocks on demand,
#   • save(...), from_checkpoint(...) — checkpoint and restore (config + weights).
#
# Philosophy:
#   • “Lazy everything” — don’t construct a model until you need one.
#   • “Small surface”   — keep this file stdlib+torch only; no heavy imports at top‑level.
#   • “Safe defaults”   — reasonable bounds and calm behavior for first‑run demos.

from __future__ import annotations

import os
from typing import Optional, Dict, Any

import torch

from .train import train_loop              # default training loop
from .infer import infer_loop              # greedy/sampling decode (+ optional relax)
from .config import Config                 # typed configuration
from .utils.logging import banner          # pretty δ⋆/β/γ banner


class Engine:
    """
    Minimal orchestration for ElementFold.

    Typical usage:
        eng = Engine(Config())              # no model yet (lazy)
        out = eng.infer()                   # → materializes model if needed, then decodes
        eng.save("ckpt.pt")                 # → {'state','cfg'} portable file
        eng2 = Engine.from_checkpoint("ckpt.pt")
        txt = eng2.steer("gentle, coherent", modality="language")

    Notes:
      • This class keeps *no training hyperparam logic*; that lives in Config/train_loop.
      • The model architecture is reconstructed inside _materialize_model() on demand.
    """

    def __init__(self, cfg: Optional[Config] = None, *, verbose: bool = False):
        self.cfg: Config = cfg or Config()
        self.model: Optional[torch.nn.Module] = None
        self._pending_state: Optional[Dict[str, Any]] = None  # weights to be loaded lazily
        # Be quiet by default; CLI (Studio) prints its own banner.
        if verbose:
            print(banner(self.cfg.delta, 1.0, 0.5))

    # ──────────────────────────────────────────────────────────────────────
    # Construction helpers
    # ──────────────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: Optional[str] = None, cfg: Optional[Config] = None) -> "Engine":
        """
        Build an Engine from a JSON config file or an existing Config.
        We do NOT load weights here — only configuration.
        """
        if cfg is None:
            if path is None:
                cfg = Config()
            else:
                with open(path, "r") as f:
                    cfg = Config.from_json(f.read())
        return cls(cfg)

    @classmethod
    def from_checkpoint(cls, ckpt_path: str) -> "Engine":
        """
        Restore an Engine from a checkpoint saved by Engine.save().
        Loads the Config snapshot immediately; defers weight materialization.
        """
        payload = torch.load(ckpt_path, map_location="cpu")  # {'state': dict|None, 'cfg': dict}
        cfg = Config(**payload.get("cfg", {}))
        eng = cls(cfg)
        eng._pending_state = payload.get("state", None)      # model comes alive on first use
        return eng

    # ──────────────────────────────────────────────────────────────────────
    # Core actions
    # ──────────────────────────────────────────────────────────────────────

    def fit(self) -> torch.nn.Module:
        """
        Train a fresh model using the project training loop.
        Any pending checkpoint weights are discarded (training supersedes them).
        """
        self.model = train_loop(**self.cfg.to_kwargs())
        if self._pending_state is not None:
            self._pending_state = None
        return self.model

    def infer(
        self,
        x: Optional[torch.Tensor] = None,
        *,
        strategy: str = "greedy",
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        relax: Optional[Dict[str, Any]] = None,   # NEW: optional diffusion/decay knobs
    ) -> Dict[str, Any]:
        """
        Run an inference pass. If no model is present, materialize it (from a pending
        checkpoint if available, otherwise by training a fresh model).

        Args:
            x: Optional token batch of shape (B, L). If None, a random (1, seq_len) batch
               is generated for quick sampling demos.
            strategy: 'greedy' or 'sample'.
            temperature, top_k, top_p: sampling knobs when strategy='sample'.
            relax: optional dict to enable the “relaxation clock” (see infer.infer_loop):
                   • computes a fold counter ℱ from the ledger (sequence‑wise),
                   • optionally diffuses the ledger once,
                   • when sampling, raises T per position: T_eff = T · exp(ρ·ℱ).

        Returns:
            A dict produced by infer_loop (e.g., {'tokens','ledger', ...} and optionally
            {'folds','relax_meta'} if 'relax' is provided).
        """
        self._ensure_model()

        # Safe default for first‑time demos: random prompt if none is provided.
        if x is None:
            device = self.device
            x = torch.randint(0, self.cfg.vocab, (1, self.cfg.seq_len), device=device)

        return infer_loop(
            self.model,
            x,
            strategy=strategy,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            relax=relax,  # ← pass through to decoding
        )

    def steer(self, prompt: Optional[str] = None, modality: str = "language") -> Any:
        """
        Convert a human prompt into control/style (β, γ, ⛔, style₅) and run the
        requested modality adapter.

        Returns:
            Whatever the adapter runner returns (often a str for textual adapters).
        """
        # Import on demand to keep Engine lightweight when used for pure train/infer.
        from .experience.steering import SteeringController
        from .experience.adapters.base import AdapterRegistry

        self._ensure_model()

        ctrl = SteeringController.load_default(self.cfg.delta)      # δ⋆‑aware controller (untrained by default)
        v = ctrl(prompt or "")                                      # ℝ⁸ = [β̂, γ̂, ⛔̂, style₅]
        factory = AdapterRegistry.get(modality)                     # fetch adapter factory
        runner = factory()                                          # produce a stateful runner
        return runner(self.model, prompt or "", v)                  # execute adapter

    # ──────────────────────────────────────────────────────────────────────
    # Steering hook for adapters (optional but handy)
    # ──────────────────────────────────────────────────────────────────────

    def apply_control(
        self,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        clamp: Optional[float] = None,
    ) -> None:
        """
        Push (β exposure, γ damping, ⛔ clamp) down into Fold–Gate–Norm blocks,
        if the model exposes a convenient hook.

        Adapters may call `model.apply_control(...)` directly; this is a friendly alias.
        """
        if self.model is None:
            return

        # Prefer a model‑level hook for cleanliness.
        if hasattr(self.model, "apply_control") and callable(self.model.apply_control):
            self.model.apply_control(beta=beta, gamma=gamma, clamp=clamp)
            return

        # Fallback: try a common structure (FGN blocks). Silently skip if unknown.
        try:
            for b in getattr(self.model, "blocks", []):
                if beta is not None and hasattr(b, "gate") and hasattr(getattr(b, "gate"), "beta"):
                    b.gate.beta.data.fill_(float(beta))
                if clamp is not None and hasattr(b, "gate") and hasattr(getattr(b, "gate"), "clamp"):
                    b.gate.clamp.data.fill_(float(clamp))
                if gamma is not None and hasattr(b, "norm") and hasattr(getattr(b, "norm"), "gamma"):
                    b.norm.gamma.data.fill_(float(gamma))
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────────────
    # Checkpointing
    # ──────────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Save a single portable file containing:
          • 'state' — model.state_dict() or None if no model yet,
          • 'cfg'   — a JSON‑serializable snapshot of Config (via .to_kwargs()).
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = self.model.state_dict() if self.model is not None else None
        torch.save({"state": state, "cfg": self.cfg.to_kwargs()}, path)

    # ──────────────────────────────────────────────────────────────────────
    # Internals (small helpers)
    # ──────────────────────────────────────────────────────────────────────

    def _ensure_model(self) -> None:
        """
        Ensure `self.model` is ready:
          • if pending weights exist → build arch + load state (lazy),
          • else if model is None    → train a fresh one,
          • else                      → no‑op.
        """
        if self.model is not None:
            return
        if self._pending_state is not None:
            self._materialize_model()
            return
        self.fit()

    def _materialize_model(self) -> None:
        """
        Build the architecture and load pending weights (if any).
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Local import avoids circulars during package import.
        from .model import Model

        m = Model(
            vocab=self.cfg.vocab,
            d=self.cfg.d,
            layers=self.cfg.layers,
            heads=self.cfg.heads,
            seq_len=self.cfg.seq_len,
            fold=self.cfg.fold,
            delta=self.cfg.delta,
        ).to(device)

        if self._pending_state is not None:
            # strict=False tolerates harmless shape drift between versions.
            m.load_state_dict(self._pending_state, strict=False)

        self.model = m
        self._pending_state = None

    @property
    def device(self) -> torch.device:
        """
        The device where the model (or the Engine’s default) lives.
        """
        if self.model is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")  # rare: model with no parameters

    def __repr__(self) -> str:
        have = "yes" if self.model is not None else ("pending" if self._pending_state is not None else "no")
        return (f"Engine(vocab={self.cfg.vocab}, d={self.cfg.d}, layers={self.cfg.layers}, "
                f"seq_len={self.cfg.seq_len}, model={have})")
