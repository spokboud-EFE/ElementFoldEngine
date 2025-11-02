# ElementFold · verify.py
# Click‑geometry self‑checks and diagnostics you can run on any ledger field X.
# Read this like a checklist:
#   1) kappa, p_half          — global coherence + half‑click boundary touch rate
#   2) seat_residual, block_residual
#                              — how far we are from perfect seat spacing and block spacing
#   3) half_click_margin      — per‑sample stability margin against rung flips
#   4) roots_of_unity_checks  — algebraic sanity for capacities (closure and harmonic isolation)

from __future__ import annotations
import math
from typing import Dict, Any, Tuple
import torch
from .ledger import phase, rung_residual  # δ⋆ phase • ↻ rung/residual


# ———————————————————————————————————————————————————————————————
# 1) Global coherence metrics (κ and p½)
# ———————————————————————————————————————————————————————————————

@torch.no_grad()
def coherence_metrics(X: torch.Tensor, delta: float, eps: float = 1e-9) -> Dict[str, float]:
    """
    Compute:
      κ      = |⟨e^{i·2πX/δ⋆}⟩|   (phase concentration; 1 = tightly locked)
      p_half = P(|r| ≥ δ⋆/2 − ε)  (fraction touching half‑click boundary)

    Accepts X with shape (B,T) or (B,) or (T,). We average over T if present.
    """
    x = X.mean(dim=1) if X.dim() == 2 else X           # reduce time dimension if needed
    ph = phase(x, delta)                                # complex phase on S¹
    kappa = float(torch.abs(ph.mean()).item())          # concentration ∈ [0,1]
    _, r = rung_residual(x, delta)                      # residual relative to nearest click center
    p_half = float((r.abs() >= (delta / 2 - eps)).float().mean().item())
    return {"kappa": kappa, "p_half": p_half}


# ———————————————————————————————————————————————————————————————
# 2) Seat / block residuals (equal‑spacing checks)
# ———————————————————————————————————————————————————————————————

@torch.no_grad()
def seat_block_residuals(X: torch.Tensor, delta: float, capacities) -> Dict[str, float]:
    """
    Check how well seats are spaced by δ⋆/C and how well blocks are spaced by δ⋆.

    Conventions:
      • X has shape (B, T) with B blocks and T seats per block (or more).
      • capacities can be an int, 1‑D tensor/list per block, or a tuple.

    Returns squared residuals (means), so smaller is better and 0 is exact.
    """
    if X.dim() == 1:
        X = X.unsqueeze(0)                              # (1,T)
    B, T = X.shape
    caps = torch.as_tensor(capacities, device=X.device, dtype=torch.long)
    if caps.numel() == 1:
        caps = caps.expand(B)
    elif caps.numel() != B:
        caps = caps[:1].expand(B)                       # fallback: broadcast first capacity

    # Seat residual: for each block, enforce ΔX ≈ δ⋆/C over the first C seats (cyclic)
    seat_err = []
    for b in range(B):
        C = int(max(1, min(int(caps[b].item()), T)))
        step = delta / float(C)
        Xb = X[b, :C]
        dif = Xb.roll(-1, dims=0) - Xb - step          # (C,)
        seat_err.append((dif.pow(2).mean()).item())
    seat_mse = float(sum(seat_err) / max(1, len(seat_err)))

    # Block residual: across blocks, enforce X[b+1,0] − X[b,0] ≈ δ⋆
    if B > 1:
        b0 = X[:, 0]
        block_dif = b0[1:] - b0[:-1] - delta           # (B-1,)
        block_mse = float(block_dif.pow(2).mean().item())
    else:
        block_mse = 0.0

    return {"seat_mse": seat_mse, "block_mse": block_mse}


# ———————————————————————————————————————————————————————————————
# 3) Half‑click margin (per‑sample stability certificate)
# ———————————————————————————————————————————————————————————————

@torch.no_grad()
def half_click_margin(x: torch.Tensor, delta: float) -> torch.Tensor:
    """
    For each scalar x, compute m = δ⋆/2 − |r| where x = k·δ⋆ + r, r ∈ (−δ⋆/2, δ⋆/2].
    Positive m certifies that the rung index is stable to perturbations < m.
    """
    _, r = rung_residual(x, delta)
    return (delta * 0.5) - r.abs()


# ———————————————————————————————————————————————————————————————
# 4) Roots of unity checks (algebraic closure)
# ———————————————————————————————————————————————————————————————

@torch.no_grad()
def roots_of_unity_checks(C: int) -> Dict[str, Any]:
    """
    Basic algebraic identities on C‑th roots of unity:
      • ∑_{a=0}^{C-1} ζ^a = 0
      • (1/C)∑ ζ^{m·a} = 1 if m≡0 mod C else 0

    We verify these numerically (float) and return small residuals.
    """
    C = int(max(1, C))
    a = torch.arange(C, dtype=torch.float64)
    z = torch.exp(2j * math.pi * a / float(C))          # ζ^a on the unit circle (complex128)
    s0 = z.sum()                                        # should be 0
    # Second identity for a handful of m values:
    ms = torch.tensor([0, 1, C - 1, C, 2 * C], dtype=torch.int64)
    residues = {}
    for m in ms.tolist():
        zm = torch.exp(2j * math.pi * (m * a) / float(C))
        mean = (zm.mean())                              # (1/C)∑ ζ^{m a}
        # Target: 1 when m % C == 0 else 0
        target = 1.0 if (m % C == 0) else 0.0
        residues[f"m={m}"] = complex(mean.item() - target)  # complex residual
    return {"sum_roots": complex(s0.item()), "avg_residues": residues}


# ———————————————————————————————————————————————————————————————
# 5) One‑stop diagnostic helper
# ———————————————————————————————————————————————————————————————

@torch.no_grad()
def diagnostics(X: torch.Tensor, delta: float, capacities) -> Dict[str, Any]:
    """
    Combine the main health signals:
      κ, p_half, seat_mse, block_mse
    """
    coh = coherence_metrics(X, delta)
    res = seat_block_residuals(X, delta, capacities)
    out = {}
    out.update(coh)
    out.update(res)
    return out

if __name__ == "__main__":
    import argparse, torch
    from .config import Config
    from .runtime import Engine

    p = argparse.ArgumentParser(description="Quick ledger diagnostics")
    p.add_argument("--steps", type=int, default=50, help="Fast micro-train before measuring")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--capacities", type=int, nargs="+", default=[2,6,10,14])
    args = p.parse_args()

    # quick micro-train (keeps it lightweight)
    cfg = Config(steps=args.steps, seq_len=args.seq_len)
    eng = Engine(cfg)
    model = eng.fit()

    # measure ledger on a single random batch
    x = torch.randint(0, model.vocab, (1, model.seq_len))
    with torch.no_grad():
        _, X = model(x)

    from .verify import diagnostics
    d = diagnostics(X, delta=cfg.delta, capacities=args.capacities)
    import json
    print(json.dumps(d, indent=2))
