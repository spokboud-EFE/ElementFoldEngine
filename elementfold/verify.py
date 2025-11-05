# ElementFold · verify.py
# Click‑geometry self‑checks and diagnostics you can run on any ledger field X.
# Read this like a checklist:
#   1) Coherence (κ, p½, margins, residual spread) — “Are we in tune?”
#   2) Equal‑spacing residuals (seat / block)      — “Are seats and blocks spaced by δ⋆/C and δ⋆?”
#   3) Half‑click margin per sample                 — “How much room before a rung flip?”
#   4) Roots‑of‑unity probes                        — “Do capacities behave algebraically on the circle?”
#
# Notes:
#   • This module is intentionally light (stdlib + torch + local geometry).
#   • We reuse telemetry.measure() so numbers match training/status prints exactly.
#   • Shapes: X may be (B,), (T,), or (B,T); we reduce time with a mean when helpful.

from __future__ import annotations

import math
from typing import Dict, Any

import torch

# Geometry + telemetry (kept as single‑source‑of‑truth)
from .ledger import phase, rung_residual, half_click_margin  # exact circular tools
from .telemetry import measure as _telemetry_measure         # κ, p_half, margins, step stats


# ──────────────────────────────────────────────────────────────────────────────
# 1) Global coherence metrics (κ, p½, margins, residual spread)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def coherence_metrics(
    X: torch.Tensor,
    delta: float,
    detail: bool = False,
) -> Dict[str, float]:
    """
    Compact coherence report over X on the δ⋆ circle.

    Args:
        X:     ledger tensor — (B,), (T,), or (B,T). If (B,T), time is averaged per sample.
        delta: click size δ⋆ (float).
        detail: if True and X is (B,T), include step statistics along T.

    Returns (all scalars):
        {
          'kappa':       phase concentration in [0,1] (1 ≡ tightly locked),
          'p_half':      boundary contact rate (fraction touching ≥½δ⋆),
          'margin_mean': average safety gap to boundary (larger is safer),
          'margin_min':  worst safety gap (can be negative if already over),
          'resid_std':   standard deviation of residuals within clicks,
          'phase_mean':  mean phase location on [0, δ⋆),
          # if detail & X is (B,T):
          'step_mean', 'step_std', 'B', 'T'
        }
    """
    m = _telemetry_measure(X, float(delta), detail=bool(detail))
    # Keep only the numerically stable scalars we want to surface
    out = {
        "kappa": float(m["kappa"]),
        "p_half": float(m["p_half"]),
        "margin_mean": float(m["margin_mean"]),
        "margin_min": float(m["margin_min"]),
        "resid_std": float(m["resid_std"]),
        "phase_mean": float(m["phase_mean"]),
    }
    if detail and "step_mean" in m:
        out.update({
            "step_mean": float(m["step_mean"]),
            "step_std": float(m["step_std"]),
            "B": int(m["B"]),
            "T": int(m["T"]),
        })
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 2) Equal‑spacing checks (seat / block residuals)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def seat_block_residuals(
    X: torch.Tensor,
    delta: float,
    capacities,
) -> Dict[str, float]:
    """
    Check how well seats are spaced by δ⋆/C and how well blocks are spaced by δ⋆.

    Conventions:
      • X has shape (B, T): B blocks (rows), T seats (cols). If X is 1‑D, we treat it as (1,T).
      • 'capacities' can be an int, a list/tuple, or a 1‑D tensor of length B.
      • We only evaluate the first C_b seats for block b (cyclic seat spacing).
      • Outputs are mean‑squared residuals (smaller is better; 0 is perfect).

    Returns:
        {'seat_mse': float, 'block_mse': float}
    """
    if X.dim() == 1:
        X = X.unsqueeze(0)  # (1,T)
    B, T = X.shape

    caps = torch.as_tensor(capacities, device=X.device, dtype=torch.long)
    if caps.numel() == 1:
        caps = caps.expand(B)
    elif caps.numel() != B:
        # Fallback: broadcast the first capacity to all blocks for a stable diagnostic.
        caps = caps[:1].expand(B)
    caps = caps.clamp(min=1, max=T)

    seat_err = []
    for b in range(B):
        Cb = int(caps[b].item())
        step = float(delta) / float(Cb)
        Xb = X[b, :Cb]                                  # first Cb seats in block b
        dif = Xb.roll(shifts=-1, dims=0) - Xb - step    # cyclic residuals vs δ⋆/C_b
        seat_err.append(dif.pow(2).mean().item())
    seat_mse = float(sum(seat_err) / max(1, len(seat_err)))

    if B > 1:
        b0 = X[:, 0]
        block_dif = b0[1:] - b0[:-1] - float(delta)     # inter‑block should be exactly δ⋆
        block_mse = float(block_dif.pow(2).mean().item())
    else:
        block_mse = 0.0

    return {"seat_mse": seat_mse, "block_mse": block_mse}


# ──────────────────────────────────────────────────────────────────────────────
# 3) Half‑click margin (per‑sample stability certificate)
# ──────────────────────────────────────────────────────────────────────────────
# Re‑exported from ledger.half_click_margin for convenience.
# Signature:
#   half_click_margin(x: Tensor, delta: float) -> Tensor
# Returns m = δ⋆/2 − |r| (positive ⇒ stable under perturbations < m).


# ──────────────────────────────────────────────────────────────────────────────
# 4) Roots‑of‑unity probes (closure & harmonic isolation for capacities)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def roots_of_unity_checks(C: int) -> Dict[str, Any]:
    """
    Algebraic identities on C‑th roots of unity ζ = e^{2πi/C}:

      • ∑_{a=0}^{C-1} ζ^a = 0                                  (closure)
      • (1/C) ∑_{a=0}^{C-1} ζ^{m·a} = 1 if m≡0 mod C else 0   (harmonic isolation)

    We verify these numerically (complex128) and return small complex residuals.
    """
    C = int(max(1, C))
    a = torch.arange(C, dtype=torch.float64)
    z = torch.exp(2j * math.pi * a / float(C))            # ζ^a ∈ ℂ
    s0 = z.sum()                                          # should be 0

    # Test a handful of m values representative of the cases.
    ms = [0, 1, C - 1, C, 2 * C]
    residues: Dict[str, complex] = {}
    for m in ms:
        zm = torch.exp(2j * math.pi * (m * a) / float(C)) # ζ^{m·a}
        mean = zm.mean()                                   # (1/C)∑ ζ^{m·a}
        target = 1.0 if (m % C == 0) else 0.0
        residues[f"m={m}"] = complex(mean.item() - target)

    return {"sum_roots": complex(s0.item()), "avg_residues": residues}


# ──────────────────────────────────────────────────────────────────────────────
# 5) One‑stop diagnostic helper
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def diagnostics(
    X: torch.Tensor,
    delta: float,
    capacities,
    detail: bool = False,
) -> Dict[str, Any]:
    """
    Combine the main health signals into one dict:

        κ, p_half, margin_mean, margin_min, resid_std, phase_mean,
        seat_mse, block_mse,
        (optional) step_mean, step_std, B, T

    The intent is to mirror the *meaning* of training logs in a static probe.
    """
    coh = coherence_metrics(X, delta, detail=detail)
    res = seat_block_residuals(X, delta, capacities)
    out: Dict[str, Any] = {}
    out.update(coh)
    out.update(res)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 6) CLI probe: quick micro‑train, then report diagnostics on a random batch
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json

    from .config import Config
    from .runtime import Engine

    ap = argparse.ArgumentParser(description="ElementFold • ledger diagnostics")
    ap.add_argument("--steps", type=int, default=50, help="micro‑train steps before measuring")
    ap.add_argument("--seq_len", type=int, default=128, help="sequence length during the probe")
    ap.add_argument("--capacities", type=int, nargs="+", default=[2, 6, 10, 14], help="seat capacities per block")
    ap.add_argument("--detail", action="store_true", help="include step statistics if X is (B,T)")
    args = ap.parse_args()

    # 1) Build config & engine; run a short fit to materialize a model.
    cfg = Config(steps=int(args.steps), seq_len=int(args.seq_len))
    eng = Engine(cfg)
    model = eng.fit()

    # 2) Grab a random token batch on the right device and get ledger coordinates X.
    device = next(model.parameters()).device
    x = torch.randint(0, model.vocab, (1, model.seq_len), device=device)
    with torch.no_grad():
        _, X = model(x)  # (B=1, T)

    # 3) Report diagnostics as JSON (stable for scripts/dashboards).
    report = diagnostics(X.detach().cpu(), delta=cfg.delta, capacities=args.capacities, detail=bool(args.detail))
    print(json.dumps(report, ensure_ascii=False, indent=2))
