# ElementFold · runtime.py
# The Engine is a small “orchestration spine” that gives you one place to:
#   • fit()     — train a model using the project’s default training loop,
#   • infer()   — run an inference pass (optionally auto‑training if the model is missing),
#   • steer()   — route a human intent (prompt) through a steering controller + adapter,
#   • save()    — checkpoint weights + the config snapshot,
#   • from_checkpoint() — restore config and (lazily) weights,
#   • apply_control()   — push (β, γ, ⛔ clamp) down into model blocks on demand.

from __future__ import annotations                         # ↻ forward annotations (older Python)
import json, os, torch                                     # ✴ JSON for cfg • OS for paths • torch for tensors/ckpt
from .train import train_loop                              # ⟲ default training loop (used by fit() and lazy train)
from .infer import infer_loop                              # ✴ inference utility (we’ll extend to sampling next)
from .config import Config                                 # ✴ typed configuration carrier
from .utils.logging import banner                          # 🄱 pretty banner (δ⋆, β, γ)


class Engine:
    def __init__(self, cfg: Config | None = None):         # ✴ construct an Engine with a config
        self.cfg = cfg or Config()                         # ≡ default config if none supplied
        self.model = None                                  # ∅ no model yet (lazy to keep boot instant)
        print(banner(self.cfg.delta, 1.0, 0.5))            # 🄱 show δ⋆ with nominal β,γ for quick sanity

    # ——————————————————————————————————————————
    # Construction helpers
    # ——————————————————————————————————————————

    @classmethod
    def load(cls, path: str | None = None, cfg: Config | None = None) -> "Engine":
        """
        Build an Engine from a JSON config file or an existing Config.
        We do NOT load weights here — only configuration.
        """
        if cfg is None:                                    # prefer explicit cfg if provided
            if path is None:                               # no file → default config
                cfg = Config()                             # safe defaults
            else:
                with open(path, "r") as f:                 # read JSON text
                    cfg = Config.from_json(f.read())       # parse into typed Config
        return cls(cfg)                                    # Engine with cfg, no model yet

    @classmethod
    def from_checkpoint(cls, ckpt_path: str) -> "Engine":
        """
        Build an Engine from a checkpoint created by Engine.save().
        We restore the config snapshot and keep weights “pending” to load lazily.
        """
        payload = torch.load(ckpt_path, map_location="cpu")  # 📖 read file: {'state': ..., 'cfg': ...}
        cfg = Config(**payload.get("cfg", {}))               # rebuild Config from stored kwargs
        eng = cls(cfg)                                       # Engine with that config
        eng._pending_state = payload.get("state", None)      # stash weights; loaded on first use
        return eng                                           # ↤ ready Engine (no model materialized yet)

    # ——————————————————————————————————————————
    # Core actions
    # ——————————————————————————————————————————

    def fit(self):                                          # ✴ train a model to completion
        self.model = train_loop(**self.cfg.to_kwargs())     # hand knobs into the training loop
        if hasattr(self, "_pending_state"):                 # if a pending ckpt existed …
            delattr(self, "_pending_state")                 # … training supersedes it
        return self.model                                   # ↤ trained model

    def infer(self, x=None, **decode):                      # ✴ run inference (lazy‑train/restore if needed)
        if self.model is None:                              # no model yet?
            if hasattr(self, "_pending_state") and self._pending_state is not None:
                self._materialize_model()                   # build arch + load weights lazily
            else:
                self.fit()                                  # otherwise, train a fresh model
        return infer_loop(self.model, x, **decode)          # delegate to inference utility (greedy/sampling)

    def steer(self, prompt: str | None = None, modality: str = "language"):
        """
        Convert a human prompt into a control/style vector and run the modality adapter.
        UX path: intent → (β,γ,⛔,style) → adapter(model, prompt, style).
        """
        # Import on demand so Engine remains light for pure train/infer flows.
        from .experience.steering import SteeringController
        from .experience.adapters.base import AdapterRegistry

        if self.model is None:                              # ensure we have a model to steer
            if hasattr(self, "_pending_state") and self._pending_state is not None:
                self._materialize_model()
            else:
                self.fit()

        ctrl = SteeringController.load_default(self.cfg.delta)   # δ⋆‑aware controller (baseline)
        adapter = AdapterRegistry.get(modality)                  # fetch modality factory
        style = ctrl(prompt or "")                               # ℝ⁸ control vector (β̂,γ̂,⛔̂,style₅)
        return adapter()(self.model, prompt or "", style)        # run adapter and return output

    # ——————————————————————————————————————————
    # Steering hook for adapters (optional but handy)
    # ——————————————————————————————————————————

    def apply_control(self, beta: float | None = None, gamma: float | None = None, clamp: float | None = None):
        """
        Push (β exposure, γ damping, ⛔ clamp) down into all Fold–Gate–Norm blocks if the model supports it.
        Adapters may call `model.apply_control(...)` directly; we expose the same hook here for convenience.
        """
        if self.model is None:
            return
        # Prefer a model‑level hook if it exists (clean separation of concerns).
        if hasattr(self.model, "apply_control") and callable(self.model.apply_control):
            self.model.apply_control(beta=beta, gamma=gamma, clamp=clamp)
            return
        # Fallback: reach into known block structure (FGNBlock) and set parameters.
        try:
            for b in getattr(self.model, "blocks", []):     # walk FGN blocks
                if beta is not None and hasattr(b.gate, "beta"):
                    b.gate.beta.data.fill_(float(beta))     # set exposure
                if clamp is not None and hasattr(b.gate, "clamp"):
                    b.gate.clamp.data.fill_(float(clamp))   # set gate clamp range
                if gamma is not None and hasattr(b.norm, "gamma"):
                    b.norm.gamma.data.fill_(float(gamma))   # set damping
        except Exception:
            pass                                            # if structure differs, silently skip

    # ——————————————————————————————————————————
    # Checkpointing
    # ——————————————————————————————————————————

    def save(self, path: str):                              # ✴ save weights + config snapshot
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)      # ensure folder exists
        state = (self.model.state_dict() if self.model is not None else None)  # weights or None
        torch.save({"state": state, "cfg": self.cfg.to_kwargs()}, path)        # single portable file

    # ——————————————————————————————————————————
    # Internals (small helpers)
    # ——————————————————————————————————————————

    def _materialize_model(self):                           # ✴ build arch and load pending weights
        device = "cuda" if torch.cuda.is_available() else "cpu"       # follow project convention
        from .model import Model                                        # local import avoids circulars at import‑time
        m = Model(                                                       # reconstruct the architecture
            vocab=self.cfg.vocab,
            d=self.cfg.d,
            layers=self.cfg.layers,
            heads=self.cfg.heads,
            seq_len=self.cfg.seq_len,
            fold=self.cfg.fold,
            delta=self.cfg.delta,
        ).to(device)
        state = getattr(self, "_pending_state", None)                   # pending state dict (or None)
        if state is not None:
            m.load_state_dict(state, strict=False)                      # load weights; strict=False tolerates shape drift
        self.model = m                                                  # attach the model
        if hasattr(self, "_pending_state"):
            delattr(self, "_pending_state")                             # clear pending state now that it’s applied

    @property
    def device(self):                                                   # ✴ convenient device accessor
        if self.model is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")                                  # rare: model with no parameters

    def __repr__(self):                                                 # ✴ friendly debug string
        have = "yes" if self.model is not None else ("pending" if hasattr(self, "_pending_state") else "no")
        return (f"Engine(vocab={self.cfg.vocab}, d={self.cfg.d}, layers={self.cfg.layers}, "
                f"seq_len={self.cfg.seq_len}, model={have})")
