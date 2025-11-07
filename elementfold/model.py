# ElementFold · model.py
# ============================================================
# ElementFold Model — from tokens to ledger.
#
# Architecture narrative
# ----------------------
# 1) RotaryClick — deterministic rotor that gives each time step a phase advance
#    proportional to δ★.  It sets the underlying temporal rhythm.
# 2) FGN stack — Fold–Gate–Norm layers that handle coherence control:
#       • fold → blend contextual features,
#       • gate → expose novelty (strength β, limited by ⛔),
#       • norm → calm energy (damping γ).
# 3) Heads — two simple projections:
#       • lm head    → token logits,
#       • ledger head → scalar X for ledger tracking.
#
# Forward contract:
#       (B,T) int64 → logits:(B,T,V), X:(B,T)
# Control contract:
#       apply_control(beta=?, gamma=?, clamp=?)
#
# Gauge:
#   δ★ defines the click size — the unit step for rotary phase
#   and for ledger regularizers.
# ============================================================

from __future__ import annotations
import math, torch
import torch.nn as nn
from typing import Tuple

from .fgn import FGNBlock  # Fold–Gate–Norm unit

# ============================================================
# 1. RotaryClick — deterministic per-time phase rotation
# ============================================================

class RotaryClick(nn.Module):
    """
    Deterministic rotation across feature lanes:

        [A';B'] = [cos θ_t  −sin θ_t][A;B]
                  [sin θ_t   cos θ_t]

    with θ_t = t·(2π·δ★).
    """
    def __init__(self, dim: int, delta: float = 0.03):
        super().__init__()
        self.dim = int(dim)
        self.theta = 2.0 * math.pi * float(delta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,T,D)
        returns (B,T,D) rotated in feature pairs.
        """
        B,T,D = x.shape
        if D < 2: return x
        even = (D//2)*2
        i0 = torch.arange(0,even,2,device=x.device)
        i1 = torch.arange(1,even,2,device=x.device)
        # per-step angle in float32 for stable trig
        ang = torch.arange(T,dtype=torch.float32,device=x.device)*self.theta
        c = torch.cos(ang).to(x.dtype).view(1,T,1)
        s = torch.sin(ang).to(x.dtype).view(1,T,1)
        A,Bp = x[:,:,i0], x[:,:,i1]
        Ar,Br = A*c - Bp*s, A*s + Bp*c
        out = x.new_empty(B,T,D)
        out[:,:,0:even:2], out[:,:,1:even:2] = Ar,Br
        if D>even: out[:,:,even:] = x[:,:,even:]
        return out

# ============================================================
# 2. Model — end-to-end ElementFold core
# ============================================================

class Model(nn.Module):
    """
    Stem → RotaryClick → [FGN]×L → LayerNorm → {lm head, ledger head}

    Outputs:
      logits : (B,T,V)
      X      : (B,T)
    """
    def __init__(self,
                 vocab:int=256, d:int=128,
                 layers:int=4, heads:int=4,
                 seq_len:int=128, fold:str="grid",
                 delta:float=0.03):
        super().__init__()
        # expose attributes for other modules
        self.vocab,self.d,self.layers,self.heads = int(vocab),int(d),int(layers),int(heads)
        self.seq_len,self.fold,self.delta = int(seq_len),str(fold),float(delta)

        # --- Stem ---
        self.emb = nn.Embedding(self.vocab,self.d)
        self.rot = RotaryClick(self.d,self.delta)

        # --- Core ---
        self.blocks = nn.ModuleList([FGNBlock(self.d) for _ in range(self.layers)])

        # --- Heads ---
        self.norm = nn.LayerNorm(self.d)
        self.lm = nn.Linear(self.d,self.vocab)   # logits
        self.ledger = nn.Linear(self.d,1)        # scalar ledger coordinate
        self._last_control = {"beta":None,"gamma":None,"clamp":None}
        self._init_weights()

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------
    def forward(self,x:torch.Tensor)->Tuple[torch.Tensor,torch.Tensor]:
        """
        Args:
            x (B,T) token ids
        Returns:
            logits (B,T,V)
            X      (B,T)
        """
        h = self.emb(x)
        h = self.rot(h)
        for blk in self.blocks: h = blk(h)
        h = self.norm(h)
        logits = self.lm(h)
        X = self.ledger(h).squeeze(-1)
        return logits,X

    # --------------------------------------------------------
    # Control interface — propagate β,γ,⛔ to all FGN blocks
    # --------------------------------------------------------
    def apply_control(self,beta=None,gamma=None,clamp=None)->None:
        """
        Pass external coherence controls into internal FGN blocks.
        Each block may define .apply_control(beta,γ,⛔); otherwise we fill known fields.
        """
        self._last_control={"beta":beta,"gamma":gamma,"clamp":clamp}
        for blk in self.blocks:
            if hasattr(blk,"apply_control") and callable(blk.apply_control):
                blk.apply_control(beta=beta,gamma=gamma,clamp=clamp); continue
            with torch.no_grad():
                gate = getattr(blk,"gate",None)
                if gate is not None:
                    if beta  is not None and hasattr(gate,"beta"): gate.beta.fill_(float(beta))
                    if clamp is not None and hasattr(gate,"clamp"): gate.clamp.fill_(float(clamp))
                norm = getattr(blk,"norm",None)
                if norm is not None and gamma is not None:
                    if hasattr(norm,"gamma"): norm.gamma.fill_(float(gamma))
                    elif hasattr(norm,"weight"): norm.weight.fill_(float(gamma))

    def last_control(self)->dict:
        """Return most recently applied β,γ,⛔ values (for dashboards)."""
        return dict(self._last_control)

    # --------------------------------------------------------
    # Initialization
    # --------------------------------------------------------
    def _init_weights(self)->None:
        nn.init.normal_(self.lm.weight,std=0.02)
        if self.lm.bias is not None: nn.init.zeros_(self.lm.bias)
        nn.init.zeros_(self.ledger.weight)
        if self.ledger.bias is not None: nn.init.zeros_(self.ledger.bias)
