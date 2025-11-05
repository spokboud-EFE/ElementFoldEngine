# ElementFold · telemetry.py
# ──────────────────────────────────────────────────────────────────────────────
# Telemetry tells us if the system is “in tune.” From ledger values X we report
# small, meaningful numbers on the δ⋆ circle:
#   • κ (kappa)     — phase alignment strength (1 = tightly aligned, 0 = spread out)
#   • p_half        — fraction touching the half‑click boundary (risk of flipping rungs)
#   • margins       — safety gap to the boundary (mean / min)
#   • residual spread — how wide residuals r are around their centers
# If X is a short sequence per sample (shape (B,T)), we can also report step stats along T.
#
# The function returns canonical ASCII keys and friendly Unicode aliases so both
# humans and code can read the same dict comfortably.

from __future__ import annotations

import math
import torch

from .ledger import phase, rung_residual, wrapped_distance  # circle tools (unit‑complex map, residuals, wrapped Δ)


def _as_tensor(X: torch.Tensor | float) -> torch.Tensor:
    """Coerce inputs to a float32 tensor (no device assumptions)."""
    return torch.as_tensor(X, dtype=torch.float32)


def _safe_item(x: torch.Tensor, default: float = 0.0) -> float:
    """Extract a Python float, falling back to a default if x is empty or non‑finite."""
    try:
        v = float(x.item() if x.numel() == 1 else x)
    except Exception:
        v = default
    if not math.isfinite(v):
        v = default
    return v


def measure(
    X: torch.Tensor,
    delta: float,
    eps: float = 1e-9,
    detail: bool = False,
) -> dict:
    """
    Compute coherence diagnostics on the δ⋆ circle.

    Args:
        X:
            Ledger values. Typical shapes:
              • (B,)   — one value per sample.
              • (B,T)  — short sequence per sample (we summarize each row).
            Other 1D/2D tensors are handled similarly; we avoid crashing on edge cases.
        delta:
            The fundamental click (δ⋆) that sets the circle size.
        eps:
            Tiny tolerance for “at the boundary” checks.
        detail:
            If True and X has shape (B,T) with T>1, also include step statistics along T.

    Returns:
        dict with scalar floats (and a few sizes if detail=True):
            {
              'kappa':      phase concentration in [0,1],
              'κ':          same as 'kappa' (Unicode alias),
              'p_half':     fraction at/over the half‑click boundary,
              'p½':         Unicode alias of 'p_half',
              'margin_mean':average safety gap to the boundary (≥0, larger is safer),
              'margin_min': smallest safety gap (can be negative → already over line),
              'resid_std':  standard deviation of residuals r,
              'phase_mean': average phase location on [0, δ⋆),
              'delta':      δ⋆ as ASCII,
              'δ⋆':          δ⋆ as Unicode,
              # if detail and X is (B,T):
              'step_mean':  average wrapped step between neighboring seats,
              'step_std':   std of those steps,
              'B': int(B), 'T': int(T)
            }
    """
    X = _as_tensor(X)
    delta = float(delta)

    # ——— 1) Reduce sequences (if any) to one representative X per sample ———
    # We accept:
    #   (B,T) → mean over T (per‑sample phase representative)
    #   (B,)  → as‑is
    #   scalar → promote to (1,)
    if X.ndim == 0:
        X = X.view(1)
    if X.ndim == 1:
        x = X
        B, T = int(X.shape[0]), 1
    elif X.ndim >= 2:
        # Only first two dims matter; extra dims (if any) are folded into B by view
        B, T = int(X.shape[0]), int(X.shape[1])
        x = X[:, :T].mean(dim=1)  # (B,)
    else:
        # Extremely unusual, but keep a safe default
        x = X.reshape(-1)
        B, T = int(x.shape[0]), 1

    # Guard: empty input
    if x.numel() == 0:
        return {
            "kappa": 0.0, "κ": 0.0,
            "p_half": 0.0, "p½": 0.0,
            "margin_mean": 0.0, "margin_min": 0.0,
            "resid_std": 0.0,
            "phase_mean": 0.0,
            "delta": delta, "δ⋆": delta,
            **({"B": 0, "T": 0} if detail else {}),
        }

    # ——— 2) Phase concentration κ on the unit circle ————————————————
    # Map x ↦ e^{i·2πx/δ⋆}; κ is the magnitude of the centroid on the unit circle.
    ph = phase(x, delta)                  # complex64/complex128 tensor on unit circle
    ph_mean = ph.mean()                   # complex scalar
    kappa = float(torch.abs(ph_mean).item())

    # Where is the mean phase located? Report in δ⋆ units wrapped into [0, δ⋆).
    ang = torch.angle(ph_mean)            # radians in (−π, π]
    phase_mean = float(((ang * delta) / (2 * math.pi)) % delta)

    # ——— 3) Residuals within a click and boundary contact rate —————————
    # x = k·δ⋆ + r, with r ∈ (−½δ⋆, ½δ⋆].
    _, r = rung_residual(x, delta)        # residuals around nearest rung
    half = delta / 2.0
    p_half = float((r.abs() >= (half - eps)).float().mean().item())

    # Safety margin to the boundary: m = ½δ⋆ − |r|
    margin = half - r.abs()
    margin_clamped = torch.clamp_min(margin, 0.0)  # don’t let negatives hide risk
    margin_mean = float(margin_clamped.mean().item())
    margin_min = float(margin.min().item())        # can be negative (over the line)

    # Residual spread (0 means everyone sits exactly on centers)
    resid_std = float(r.std(unbiased=False).item() if r.numel() > 1 else 0.0)

    report = {
        # Canonical keys
        "kappa": kappa,
        "p_half": p_half,
        "margin_mean": margin_mean,
        "margin_min": margin_min,
        "resid_std": resid_std,
        "phase_mean": phase_mean,
        "delta": delta,
        # Unicode aliases for friendly dashboards/CLIs
        "κ": kappa,
        "p½": p_half,
        "δ⋆": delta,
    }

    # ——— 4) Optional step statistics along T when sequences are provided ———
    if detail and X.ndim >= 2 and T > 1:
        # Wrapped seat‑to‑seat distance captures local “bumpiness” along the sequence dimension.
        d = wrapped_distance(X[:, 1:], X[:, :-1], delta)  # same shape as X[:, 1:]
        step_mean = float(d.mean().item())
        step_std = float(d.std(unbiased=False).item() if d.numel() > 1 else 0.0)
        report.update({"step_mean": step_mean, "step_std": step_std, "B": B, "T": T})

    return report
