# ElementFold · telemetry.py
# ──────────────────────────────────────────────────────────────────────────────
# Telemetry tells us if the system is “in tune.” From ledger values X we report
# small, meaningful numbers on the δ⋆ circle:
#   • κ (kappa)       — phase alignment strength (1 = tightly aligned, 0 = spread out)
#   • p_half          — fraction touching the half‑click boundary (risk of flipping rungs)
#   • margins         — safety gap to the boundary (mean / min)
#   • residual spread — how wide residuals r are around their centers
# If X is a short sequence per sample (shape (B,T)), we can also report step stats along T.
#
# The module returns **canonical ASCII keys** and **friendly Unicode aliases** so
# both humans and code can read the same dict comfortably. It also exposes:
#   • normalize() — maps Unicode/short human keys (κ, p½, δ⋆, x) → ASCII (kappa, p_half, delta, x_mean)
#   • pretty()    — one‑line status string for Studio/CLI dashboards
#
# Plain words:
#   We turn raw seat locations X into compact “how calm are we?” numbers. These
#   numbers drive controllers gently (β, γ, ⛔) and give the Studio something
#   easy to read: κ near 1 means “well‑locked,” p½ near 0.5 means “on the ridge.”

from __future__ import annotations

import math
from typing import Dict, Optional, TypedDict, Any

import torch

from .ledger import (
    phase,            # map x ↦ e^{i·2πx/δ⋆} on the unit circle
    rung_residual,    # x = k·δ⋆ + r  with r ∈ (−½δ⋆, ½δ⋆]
    wrapped_distance, # wrapped |Δ| along δ⋆
)

# ──────────────────────────────────────────────────────────────────────────────
# Public schema (typing aid; runtime remains plain dict[str, float])
# ──────────────────────────────────────────────────────────────────────────────

class Telemetry(TypedDict, total=False):
    # Canonical ASCII keys (preferred by code)
    kappa: float           # ∈ [0,1]
    p_half: float          # ∈ [0,1]
    margin_mean: float
    margin_min: float
    resid_std: float
    phase_mean: float      # location of mean phase in [0, δ⋆)
    delta: float           # δ⋆
    x_mean: float          # global mean of X across (B,T) if provided
    k_now: int             # nearest rung index to x_mean (estimate)
    step_mean: float       # optional (when sequences): mean wrapped step along T
    step_std: float        # optional (when sequences): std of wrapped steps
    B: int                 # batch size when sequences provided
    T: int                 # time steps when sequences provided
    # Diffusion/relaxation hooks (optional; pass-through if provided)
    eta: float             # share rate per unit path (η)
    folds: float           # accumulated folds ℱ

    # Unicode aliases (mirrors; friendly for dashboards)
    # We include them in the returned dict for readability.
    # They are kept in sync with the ASCII fields.
    #   κ ↔ kappa, p½ ↔ p_half, δ⋆ ↔ delta
    # (Typing systems ignore these non-ASCII keys; they are still valid at runtime.)
    # fmt: off
    # mypy/pyright will ignore these; they are here to document the contract.
    # fmt: on


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _as_tensor(X: torch.Tensor | float) -> torch.Tensor:
    """Coerce inputs to a float32 tensor (no device assumptions)."""
    return torch.as_tensor(X, dtype=torch.float32)


def _safe_item(x: torch.Tensor, default: float = 0.0) -> float:
    """Extract a Python float, falling back to `default` on empties or non‑finite."""
    try:
        v = float(x.item() if x.numel() == 1 else x)
    except Exception:
        v = default
    if not math.isfinite(v):
        v = default
    return v


# ──────────────────────────────────────────────────────────────────────────────
# Core measurement
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def measure(
    X: torch.Tensor,
    delta: float,
    eps: float = 1e-9,
    detail: bool = False,
) -> Telemetry:
    """
    Compute coherence diagnostics on the δ⋆ circle.

    Args
    ----
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

    Returns
    -------
    Telemetry dict with scalar floats (plus sizes if detail=True). Canonical ASCII
    keys are provided alongside Unicode aliases for human‑friendly displays.
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
        B, T = int(X.shape[0]), int(X.shape[1])
        x = X[:, :T].mean(dim=1)  # (B,)
    else:
        x = X.reshape(-1)
        B, T = int(x.shape[0]), 1

    # Guard: empty input
    if x.numel() == 0:
        return Telemetry(
            kappa=0.0, p_half=0.0,
            margin_mean=0.0, margin_min=0.0,
            resid_std=0.0, phase_mean=0.0,
            delta=delta,
            x_mean=0.0, k_now=0,
            **({"B": 0, "T": 0} if detail else {}),
            **{"κ": 0.0, "p½": 0.0, "δ⋆": delta},
        )

    # A global mean location (one number for controllers/UIs)
    x_mean = _safe_item(x.mean(), 0.0)
    k_now = int(math.floor((x_mean / delta) + 0.5))

    # ——— 2) Phase concentration κ on the unit circle ————————————————
    # Map x ↦ e^{i·2πx/δ⋆}; κ is the magnitude of the centroid on the unit circle.
    ph = phase(x, delta)          # complex64/complex128 tensor on unit circle
    ph_mean = ph.mean()           # complex scalar
    kappa = float(torch.abs(ph_mean).item())

    # Where is the mean phase located? Report in δ⋆ units wrapped into [0, δ⋆).
    ang = torch.angle(ph_mean)    # radians in (−π, π]
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

    report: Telemetry = {
        # Canonical keys for code
        "kappa": kappa,
        "p_half": p_half,
        "margin_mean": margin_mean,
        "margin_min": margin_min,
        "resid_std": resid_std,
        "phase_mean": phase_mean,
        "delta": delta,
        "x_mean": x_mean,
        "k_now": k_now,
        # Unicode aliases for friendly dashboards/CLIs (mirrors)
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


# ──────────────────────────────────────────────────────────────────────────────
# Normalization (Unicode/aliases → canonical ASCII)
# ──────────────────────────────────────────────────────────────────────────────

# We accept a few human‑friendly shorthands commonly seen in CLIs/UIs/adapters.
# These are mapped to canonical ASCII keys so controllers can rely on one schema.
_ALIAS_TABLE: Dict[str, str] = {
    # telemetry
    "κ": "kappa",
    "kappa": "kappa",
    "p½": "p_half",
    "p_half": "p_half",
    "δ⋆": "delta",
    "δ*": "delta",
    "δ": "delta",
    "delta": "delta",
    "x̄": "x_mean",
    "xbar": "x_mean",
    "x_mean": "x_mean",
    "x": "x_mean",          # adapters sometimes send 'x'
    "k": "k_now",           # if a source reports a rung index directly
    "k_now": "k_now",
    "φ̄": "phase_mean",
    "phase_mean": "phase_mean",
    "rσ": "resid_std",
    "resid_std": "resid_std",
    "m̄": "margin_mean",
    "margin_mean": "margin_mean",
    "m_min": "margin_min",
    "margin_min": "margin_min",
    # diffusion / relaxation (optional, pass-through if present)
    "η": "eta",
    "eta": "eta",
    "ℱ": "folds",
    "F": "folds",
    "folds": "folds",
    # (Not strictly telemetry, but mapping these helps when status blobs are merged)
    "β": "beta",
    "gamma": "gamma",
    "γ": "gamma",
    "⛔": "clamp",
    "clamp": "clamp",
}


def normalize(d: Dict[str, Any]) -> Telemetry:
    """
    Convert a mixed‑key telemetry dict (Unicode, shorthands) into canonical ASCII.

    Examples
    --------
    normalize({'κ':0.9, 'p½':0.02, 'δ⋆':0.5, 'x':0.13})
      → {'kappa':0.9, 'p_half':0.02, 'delta':0.5, 'x_mean':0.13, 'κ':0.9, 'p½':0.02, 'δ⋆':0.5}

    Design
    ------
    • Keeps both ASCII and Unicode mirrors; code should read ASCII, UIs can read either.
    • Best‑effort float casting; missing keys are left out rather than forced to 0.0.
    """
    if not d:
        return Telemetry()

    out: Telemetry = {}
    # First pass: map to ASCII names when we know them.
    for k, v in d.items():
        k_ascii = _ALIAS_TABLE.get(k, None)
        if k_ascii is None:
            # Keep unknown keys as‑is (they might be app‑specific extras)
            try:
                out[k] = float(v) if isinstance(v, (int, float)) else v  # type: ignore[assignment]
            except Exception:
                out[k] = v  # type: ignore[assignment]
            continue

        # Canonicalize numeric payloads
        vv: Any = v
        if isinstance(v, (int, float, bool)):
            vv = float(v) if not isinstance(v, bool) else (1.0 if v else 0.0)
        elif isinstance(v, torch.Tensor):
            vv = _safe_item(v, 0.0)

        out[k_ascii] = vv  # canonical ASCII

    # Second pass: add Unicode mirrors for the common trio (nice for dashboards)
    if "kappa" in out:
        out["κ"] = float(out["kappa"])  # type: ignore[index]
    if "p_half" in out:
        out["p½"] = float(out["p_half"])  # type: ignore[index]
    if "delta" in out:
        out["δ⋆"] = float(out["delta"])  # type: ignore[index]

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Pretty printing (Studio/CLI one‑liner)
# ──────────────────────────────────────────────────────────────────────────────

def _bar(x: float, n: int = 10) -> str:
    """
    Tiny unicode bar: ▮ for filled, ▯ for empty. Values beyond [0,1] are clamped.
    """
    x = 0.0 if not math.isfinite(x) else max(0.0, min(1.0, x))
    filled = int(round(x * n))
    return "▮" * filled + "▯" * (n - filled)


def pretty(d: Dict[str, Any], ascii: bool = False) -> str:
    """
    Human‑readable one‑liner summarizing telemetry for the Studio.

    • Uses normalize() internally (so you can pass mixed keys).
    • Shows κ and p½ with micro bars; includes δ⋆ and x̄ estimates.
    • If step stats exist, appends them.

    Examples
    --------
    pretty({'κ':0.93, 'p½':0.04, 'δ⋆':0.5, 'x':0.12})
      → 'κ 0.93 ▮▮▮▮▮▮▮▯▯▯  |  p½ 0.04 ▮▯▯▯▯▯▯▯▯▯  |  δ⋆=0.500  x̄=0.120  k≈0'
    """
    t = normalize(d)

    kappa = float(t.get("kappa", 0.0))
    p_half = float(t.get("p_half", 0.0))
    delta = float(t.get("delta", 0.0))
    x_mean = float(t.get("x_mean", 0.0))
    k_now = int(t.get("k_now", 0))
    resid = float(t.get("resid_std", 0.0))
    mmin  = float(t.get("margin_min", 0.0))

    # Optional step stats
    step_mean = t.get("step_mean", None)
    step_std  = t.get("step_std", None)

    if ascii:
        head = f"kappa {kappa:0.2f} {_bar(kappa)}  |  p_half {p_half:0.2f} {_bar(p_half)}"
        tail = f" | delta={delta:.3f}  x_mean={x_mean:.3f}  k~{k_now}  rσ={resid:.3g}  m_min={mmin:.3g}"
        if step_mean is not None and step_std is not None:
            tail += f"  step={float(step_mean):.3g}±{float(step_std):.3g}"
        return head + tail

    # Unicode flavor
    head = f"κ {kappa:0.2f} {_bar(kappa)}  |  p½ {p_half:0.2f} {_bar(p_half)}"
    tail = f"  |  δ⋆={delta:.3f}  x̄={x_mean:.3f}  k≈{k_now}  rσ={resid:.3g}  m_min={mmin:.3g}"
    if step_mean is not None and step_std is not None:
        tail += f"  step={float(step_mean):.3g}±{float(step_std):.3g}"
    return head + tail
