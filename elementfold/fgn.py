# ElementFold · fgn.py
# Fold–Gate–Norm (FGN) is the engine’s heartbeat:
#   1) Fold:      gather local structure (here along time) without blowing up scale.
#   2) Gate:      compute a scalar exposure per step and apply an exponential gain e^{β·g}.
#   3) Normalize: damp and keep the result bounded so depth can grow safely.
# Finally, we add a residual lane so identity is always available.

import torch, torch.nn as nn, torch.nn.functional as F  # We use PyTorch modules and ops throughout.


class FoldGrid(nn.Module):
    """
    Depth‑wise 1D convolution over the time axis.
    Each feature channel is folded independently (groups=d), so we never mix channels here.
    This preserves identity and keeps the fold non‑expansive by default.

    Args:
        d:      number of channels (feature dimension D).
        kind:   'identity' (center tap = 1) or 'avg3' ([1,2,1]/4 smoothing); both are safe starting points.
        learn:  if True, the kernel is learnable; if False, it stays as initialized.
    """
    def __init__(self, d: int, kind: str = "identity", learn: bool = True):
        super().__init__()                                                # Standard nn.Module init.
        self.conv = nn.Conv1d(d, d, kernel_size=3, padding=1, groups=d,
                              bias=False)                                  # Depth‑wise conv: (B,D,T) → (B,D,T)
        with torch.no_grad():                                              # Initialize to a stable, non‑expansive kernel.
            k = torch.zeros(d, 1, 3)                                       # Allocate a kernel tensor per channel.
            if kind == "avg3":                                             # Simple 3‑tap smoother.
                k[:, :, 0] = 0.25; k[:, :, 1] = 0.5; k[:, :, 2] = 0.25
            else:                                                          # 'identity': pass‑through center tap.
                k[:, :, 1] = 1.0
            self.conv.weight.copy_(k)                                      # Load weights into the conv.
        self.conv.weight.requires_grad = bool(learn)                       # Freeze or learn according to 'learn'.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,T,D) — batch, time, features
        returns y: (B,T,D) — folded features (per‑channel aggregation along time)
        """
        b, t, d = x.shape                                                  # Read shape for clarity (not strictly needed).
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)                   # Conv1d expects (B,D,T); we transpose in/out.
        return y                                                           # Return the folded tensor unchanged in shape.


class Gate(nn.Module):
    """
    Exponential gate y ← y * exp(β · g(x)), where g(x) is a learned scalar per time step.
    We center g by subtracting its row‑wise max (so max gain = 1) and clamp its range for safety.

    Args:
        d: feature dimension D (input to the small linear probe φ: ℝ^D→ℝ).
    """
    def __init__(self, d: int):
        super().__init__()                                                 # nn.Module init.
        self.lin = nn.Linear(d, 1)                                         # φ(x): one scalar gate potential per step.
        self.beta = nn.Parameter(torch.tensor(1.0))                        # β (exposure strength) — learnable knob.
        self.clamp = nn.Parameter(torch.tensor(5.0))                       # ⛔ clamp on negative side (range [−clamp,0]).

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,T,D)
        returns gexp: (B,T,1) — multiplicative gain to apply to folded features
        """
        g = self.lin(x).squeeze(-1)                                        # φ(x) → (B,T) one scalar per time step.
        g = g - g.max(dim=1, keepdim=True).values                          # Center so the largest g per sequence is 0 ⇒ max gain=1.
        g = g.clamp(min=-self.clamp.item(), max=0.0)                       # Avoid runaway gains; only attenuate or keep as‑is.
        gexp = torch.exp(self.beta * g).unsqueeze(-1)                      # Convert to multiplicative gain e^{β·g}.
        return gexp                                                        # Shape (B,T,1), broadcastable over channels.

    def set_control(self, beta: float = None, clamp: float = None):
        """
        External control hook (Supervisor can call this).
        """
        if beta is not None:
            with torch.no_grad():
                self.beta.copy_(torch.tensor(float(beta), device=self.beta.device))
        if clamp is not None:
            with torch.no_grad():
                self.clamp.copy_(torch.tensor(float(clamp), device=self.clamp.device))


class Norm(nn.Module):
    """
    Row‑wise energy normalization to keep updates bounded.
    We divide by (‖y‖₁ + ε)^γ so larger activations get damped more when γ>0.

    Args:
        d:     feature dimension (not used directly; kept for symmetry with Gate/Fold).
        gamma: damping strength in [0,1]; 0 means no damping, 1 means strong damping.
    """
    def __init__(self, d: int, gamma: float = 0.5):
        super().__init__()                                                 # nn.Module init.
        self.gamma = nn.Parameter(torch.tensor(float(gamma)))              # γ — learnable; we clamp at use time.
        self.eps = 1e-6                                                    # ε — keeps division well‑defined.

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        y: (B,T,D) — gated features
        returns y_norm: (B,T,D) — normalized features
        """
        gamma = torch.clamp(self.gamma, 0.0, 1.0)                          # Make sure γ stays in [0,1] during training.
        e = (y.abs().sum(dim=-1, keepdim=True) + self.eps).pow(gamma)      # Compute (‖y‖₁ + ε)^γ per (B,T,1).
        return y / e                                                        # Divide to damp energy while preserving shape.

    def set_control(self, gamma: float = None):
        """
        External control hook (Supervisor can call this).
        """
        if gamma is not None:
            with torch.no_grad():
                self.gamma.copy_(torch.tensor(float(gamma), device=self.gamma.device))


class FGNBlock(nn.Module):
    """
    One Fold–Gate–Norm block with a residual projection.
    Computation:
        y = Fold(x)
        y = y * Gate(x)           # multiplicative exposure
        y = Norm(y)               # energy damping
        out = x + Proj(y)         # residual identity lane

    Args:
        d:            feature dimension.
        fold_kind:    pass 'identity' (default) or 'avg3' for smoother folds.
        fold_learn:   whether the fold kernel should be trainable.
        resid_scale:  scale on the residual update before adding back to x.
        dropout:      optional dropout after projection for a touch of regularization (default 0).
    """
    def __init__(self, d: int, fold_kind: str = "identity", fold_learn: bool = True,
                 resid_scale: float = 1.0, dropout: float = 0.0):
        super().__init__()                                                 # nn.Module init.
        self.fold = FoldGrid(d, kind=fold_kind, learn=fold_learn)          # 1) local aggregation along time.
        self.gate = Gate(d)                                                # 2) exponential gate on a scalar potential.
        self.norm = Norm(d)                                                # 3) row‑wise energy damping.
        self.proj = nn.Linear(d, d)                                        # Project y back into the feature space.
        self.drop = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()  # Optional small regularizer.
        self.resid_scale = float(resid_scale)                              # Residual scaling factor (e.g., 1.0).

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,T,D)
        returns out: (B,T,D)
        """
        y = self.fold(x)                                                   # (1) collect from neighbors without mixing channels.
        g = self.gate(x)                                                   # (2) compute a per‑step gain from the current state x.
        y = y * g                                                          #     apply exposure multiplicatively.
        y = self.norm(y)                                                   # (3) keep the update bounded so depth stays stable.
        y = self.drop(self.proj(y))                                        # Project (and maybe drop) before adding back.
        return x + self.resid_scale * y                                    # Residual add: identity always has a clean lane.

    # Small convenience so the Supervisor can drive β, γ, and ⛔ (clamp) from outside.
    def apply_control(self, beta: float = None, gamma: float = None, clamp: float = None):
        """
        Update inner gate/norm hyper‑parameters from an external controller.
        """
        self.gate.set_control(beta=beta, clamp=clamp)                      # Forward β and clamp.
        self.norm.set_control(gamma=gamma)                                 # Forward γ.
