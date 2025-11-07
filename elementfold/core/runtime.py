# ElementFold · runtime.py
# ============================================================
# Runtime Engine — small orchestration spine for ElementFold.
#
# Responsibilities
# ----------------
# • fit()               → run the default training loop
# • infer(...)          → decode tokens (greedy or sample)
# • steer(...)          → route human intent via steering adapters
# • apply_control(...)  → propagate (β, γ, ⛔) to model blocks
# • save(), from_checkpoint() → checkpoint and restore
#
# Philosophy
# -----------
# “Lazy everything” — build nothing until first use.
# “Small surface”   — stdlib + torch only.
# “Safe defaults”   — calm behavior for first‑run demos.
# ============================================================

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from .config import Config  # lightweight; safe to import at module load

__all__ = ["Engine"]


class Engine:
    """
    Minimal orchestration object.

    Typical usage:
        eng = Engine(Config())
        eng.fit()           # train new model
        out = eng.infer()   # decode
        eng.save("ckpt.pt")
        eng2 = Engine.from_checkpoint("ckpt.pt")
        txt = eng2.steer("gentle, coherent")

    Notes:
      • Keeps no optimizer or loop state — just config + model.
      • Model is built lazily when first needed.
    """

    def __init__(self, cfg: Optional[Config] = None, *, verbose: bool = False) -> None:
        self.cfg: Config = cfg or Config()
        self.model: Optional[nn.Module] = None
        self._pending_state: Optional[Dict[str, torch.Tensor]] = None

        if verbose:
            # Lazy import for display utilities; fall back to a simple line if absent.
            try:
                from .utils.display import banner as _banner  # type: ignore
                print(_banner(self.cfg.delta, 1.0, 0.5))
            except Exception:
                print(f"ElementFold Engine — δ⋆={self.cfg.delta:.3f}")

    # ========================================================
    # Construction helpers
    # ========================================================

    @classmethod
    def load(cls, path: Optional[str] = None, cfg: Optional[Config] = None) -> "Engine":
        """
        Build an Engine from a JSON config file or an existing Config.
        """
        if cfg is None:
            if path is None:
                cfg = Config()
            else:
                with open(path, "r", encoding="utf-8") as f:
                    cfg = Config.from_json(f.read())
        return cls(cfg)

    @classmethod
    def from_checkpoint(cls, ckpt_path: str) -> "Engine":
        """
        Restore Engine from a checkpoint produced by Engine.save().
        Loads config immediately; defers weight materialization until first use.
        """
        payload = torch.load(ckpt_path, map_location="cpu")
        cfg_dict = payload.get("cfg", {})
        cfg = Config(**cfg_dict) if isinstance(cfg_dict, dict) else Config()
        eng = cls(cfg)
        state = payload.get("state", None)
        if isinstance(state, dict):
            eng._pending_state = state
        else:
            eng._pending_state = None
        return eng

    # ========================================================
    # Core actions
    # ========================================================

    def fit(self) -> nn.Module:
        """
        Train a model using the project's default training loop.
        Lazily imports the training routine to keep the surface small.
        """
        from .train import train_loop  # lazy
        trained = train_loop(**self.cfg.to_kwargs())
        # Accept either a model or (model, …) tuples.
        self.model = trained if isinstance(trained, nn.Module) else trained[0]
        self._pending_state = None
        return self.model

    @torch.no_grad()
    def infer(
        self,
        x: Optional[torch.Tensor] = None,
        *,
        strategy: str = "greedy",
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        relax: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Decode tokens. Builds the model on first use.
        Optionally applies relaxation controls (e.g., fold counter).
        """
        self._ensure_model()
        from .infer import infer_loop  # lazy

        if x is None:
            device = self.device
            x = torch.randint(0, self.cfg.vocab, (1, self.cfg.seq_len), device=device)

        return infer_loop(
            self.model,  # type: ignore[arg-type]
            x,
            strategy=strategy,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            relax=relax,
        )

    def steer(self, prompt: Optional[str] = None, modality: str = "language") -> Any:
        """
        Route a human prompt through a steering controller + adapter.
        """
        self._ensure_model()
        try:
            from .experience.steering import SteeringController  # type: ignore
            from .experience.adapters.base import AdapterRegistry  # type: ignore
        except Exception as e:
            raise RuntimeError("Steering stack not available in this build.") from e

        ctrl = SteeringController.load_default(self.cfg.delta)
        value = ctrl(prompt or "")
        factory = AdapterRegistry.get(modality)
        runner = factory()
        return runner(self.model, prompt or "", value)  # type: ignore[arg-type]

    # ========================================================
    # Control interface (β, γ, ⛔)
    # ========================================================

    def apply_control(
        self,
        *,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        clamp: Optional[float] = None,
    ) -> None:
        """
        Push (β, γ, ⛔) into the model or its FGN blocks.
        """
        if self.model is None:
            return

        # Preferred path: model exposes a single dispatcher.
        if hasattr(self.model, "apply_control"):
            try:
                self.model.apply_control(beta=beta, gamma=gamma, clamp=clamp)  # type: ignore[attr-defined]
                return
            except Exception:
                pass

        # Fallback path: touch known submodules if present.
        try:
            for b in getattr(self.model, "blocks", []):  # type: ignore[attr-defined]
                gate = getattr(b, "gate", None)
                norm = getattr(b, "norm", None)
                if beta is not None and getattr(getattr(gate, "beta", None), "data", None) is not None:
                    gate.beta.data.fill_(float(beta))  # type: ignore[union-attr]
                if clamp is not None and getattr(getattr(gate, "clamp", None), "data", None) is not None:
                    gate.clamp.data.fill_(float(clamp))  # type: ignore[union-attr]
                if gamma is not None and getattr(getattr(norm, "gamma", None), "data", None) is not None:
                    norm.gamma.data.fill_(float(gamma))  # type: ignore[union-attr]
        except Exception:
            # Stay silent; control is best-effort.
            pass

    # ========================================================
    # Checkpointing
    # ========================================================

    def save(self, path: str) -> None:
        """
        Save {'state','cfg'} to a single portable file.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = self.model.state_dict() if self.model is not None else None
        torch.save({"state": state, "cfg": self.cfg.to_kwargs()}, path)

    # ========================================================
    # Internals
    # ========================================================

    def _ensure_model(self) -> None:
        """
        Ensure a model exists; build or train as needed.
        """
        if self.model is not None:
            return
        if self._pending_state is not None:
            self._materialize_model()
            return
        self.fit()  # default: train from scratch if nothing is available

    def _materialize_model(self) -> None:
        """
        Build architecture and load pending weights.
        """
        from .model import Model  # lazy

        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"
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
            try:
                m.load_state_dict(self._pending_state, strict=False)
            except Exception:
                # Partial/shape-mismatched states are tolerated in 'strict=False' mode.
                pass

        self.model = m
        self._pending_state = None

    # ========================================================
    # Properties & representation
    # ========================================================

    @property
    def device(self) -> torch.device:
        """
        Return the device of the current model or a best guess if uninitialized.
        """
        if self.model is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            return next(self.model.parameters()).device  # type: ignore[call-arg]
        except StopIteration:
            return torch.device("cpu")

    def __repr__(self) -> str:
        have = "yes" if self.model else ("pending" if self._pending_state else "no")
        return (
            f"Engine(vocab={self.cfg.vocab}, d={self.cfg.d}, "
            f"layers={self.cfg.layers}, seq_len={self.cfg.seq_len}, model={have})"
        )
