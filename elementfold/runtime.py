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
# “Safe defaults”   — calm behavior for first-run demos.
# ============================================================

from __future__ import annotations
import os, torch
from typing import Optional, Dict, Any

from .train import train_loop
from .infer import infer_loop
from .config import Config
from .utils.display import banner


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

    def __init__(self, cfg: Optional[Config]=None, *, verbose:bool=False):
        self.cfg = cfg or Config()
        self.model: Optional[torch.nn.Module] = None
        self._pending_state: Optional[Dict[str,Any]] = None
        if verbose:
            print(banner(self.cfg.delta,1.0,0.5))

    # ========================================================
    # Construction helpers
    # ========================================================

    @classmethod
    def load(cls, path:Optional[str]=None, cfg:Optional[Config]=None)->Engine:
        """Load an Engine from a JSON config or an existing Config."""
        if cfg is None:
            if path is None: cfg=Config()
            else:
                with open(path,"r") as f: cfg=Config.from_json(f.read())
        return cls(cfg)

    @classmethod
    def from_checkpoint(cls, ckpt_path:str)->Engine:
        """Restore Engine from checkpoint: loads cfg, defers weight materialization."""
        payload=torch.load(ckpt_path,map_location="cpu")
        cfg=Config(**payload.get("cfg",{}))
        eng=cls(cfg)
        eng._pending_state=payload.get("state",None)
        return eng

    # ========================================================
    # Core actions
    # ========================================================

    def fit(self)->torch.nn.Module:
        """Train a model using project’s default training loop."""
        self.model=train_loop(**self.cfg.to_kwargs())
        self._pending_state=None
        return self.model

    def infer(self,x:Optional[torch.Tensor]=None,*,strategy:str="greedy",
              temperature:float=1.0,top_k:Optional[int]=None,top_p:Optional[float]=None,
              relax:Optional[Dict[str,Any]]=None)->Dict[str,Any]:
        """
        Decode tokens. Builds model on first use.
        Optionally applies relaxation (fold counter, smoothing, temperature lift).
        """
        self._ensure_model()
        if x is None:
            device=self.device
            x=torch.randint(0,self.cfg.vocab,(1,self.cfg.seq_len),device=device)
        return infer_loop(self.model,x,strategy=strategy,temperature=temperature,
                          top_k=top_k,top_p=top_p,relax=relax)

    def steer(self,prompt:Optional[str]=None,modality:str="language")->Any:
        """Route a human prompt through a steering controller + adapter."""
        from .experience.steering import SteeringController
        from .experience.adapters.base import AdapterRegistry
        self._ensure_model()
        ctrl=SteeringController.load_default(self.cfg.delta)
        v=ctrl(prompt or "")
        factory=AdapterRegistry.get(modality)
        runner=factory()
        return runner(self.model,prompt or "",v)

    # ========================================================
    # Control interface (β, γ, ⛔)
    # ========================================================

    def apply_control(self,*,beta:Optional[float]=None,
                      gamma:Optional[float]=None,clamp:Optional[float]=None)->None:
        """Push (β,γ,⛔) into model or its FGN blocks."""
        if self.model is None: return
        if hasattr(self.model,"apply_control"):
            self.model.apply_control(beta=beta,gamma=gamma,clamp=clamp)
            return
        try:
            for b in getattr(self.model,"blocks",[]):
                if beta  is not None and hasattr(b,"gate") and hasattr(b.gate,"beta"):
                    b.gate.beta.data.fill_(float(beta))
                if clamp is not None and hasattr(b,"gate") and hasattr(b.gate,"clamp"):
                    b.gate.clamp.data.fill_(float(clamp))
                if gamma is not None and hasattr(b,"norm") and hasattr(b.norm,"gamma"):
                    b.norm.gamma.data.fill_(float(gamma))
        except Exception: pass

    # ========================================================
    # Checkpointing
    # ========================================================

    def save(self,path:str)->None:
        """Save {'state','cfg'} to a single portable file."""
        os.makedirs(os.path.dirname(path) or ".",exist_ok=True)
        state=self.model.state_dict() if self.model else None
        torch.save({"state":state,"cfg":self.cfg.to_kwargs()},path)

    # ========================================================
    # Internals
    # ========================================================

    def _ensure_model(self)->None:
        """Ensure model exists; build or train as needed."""
        if self.model is not None: return
        if self._pending_state is not None: self._materialize_model(); return
        self.fit()

    def _materialize_model(self)->None:
        """Build architecture and load pending weights."""
        device="cuda" if torch.cuda.is_available() else "cpu"
        from .model import Model
        m=Model(vocab=self.cfg.vocab,d=self.cfg.d,layers=self.cfg.layers,
                heads=self.cfg.heads,seq_len=self.cfg.seq_len,
                fold=self.cfg.fold,delta=self.cfg.delta).to(device)
        if self._pending_state is not None:
            m.load_state_dict(self._pending_state,strict=False)
        self.model=m; self._pending_state=None

    # ========================================================
    # Properties & representation
    # ========================================================

    @property
    def device(self)->torch.device:
        """Return device of current model or default best guess."""
        if self.model is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try: return next(self.model.parameters()).device
        except StopIteration: return torch.device("cpu")

    def __repr__(self)->str:
        have="yes" if self.model else ("pending" if self._pending_state else "no")
        return (f"Engine(vocab={self.cfg.vocab}, d={self.cfg.d}, "
                f"layers={self.cfg.layers}, seq_len={self.cfg.seq_len}, model={have})")
