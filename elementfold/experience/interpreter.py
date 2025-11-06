# ElementFold · experience/interpreter.py
# ──────────────────────────────────────────────────────────────────────────────
# Tensor‑anchored interpreter:
#   Turn raw tensors + runtime knobs into a *human* narrative with suggestions.
#
# Why this file?
#   • We want “numbers with meaning” — every scalar gets a sentence.
#   • We want consistent text for Studio *and* the Web UI (same source of truth).
#   • We keep it portable (stdlib + torch) and tolerant of missing fields.
#
# Inputs (typical):
#   interpret(
#     X=ledger_tensor,                    # (B,T) float, optional
#     delta=cfg.delta,                    # float (δ⋆)
#     folds=out.get("folds"),             # (B,T) float from infer_loop(relax=...), optional
#     relax_meta=out.get("relax_meta"),   # dict with {'eta','lambda','D','rho','dt',...}, optional
#     control={'beta':b, 'gamma':g, 'clamp':c},  # optional
#     rung=rung_controller.status(),      # optional dict: {'intent','phase','dwell','band',...}
#     tele_hint={...},                    # optional fast path: {'kappa','p_half',...}
#   )
#
# Output (dict, JSON‑friendly):
#   {
#     'headline':  "LOCKED • κ=0.82  p½=0.11  ℱ=0.27  z=1.37  A≈e^(−2ℱ)=0.58",
#     'caption':   "β=1.28  γ=0.46  ⛔=5.0  |  far from barrier; stable and coherent",
#     'badges':    {'F':0.27, 'z':1.37, 'A':0.58},
#     'traffic':   {'coherence':'good', 'barrier':'good', 'stability':'good', 'pace':'warn'},
#     'metrics':   { 'kappa': {...}, 'p_half': {...}, 'margin_min': {...}, ... },
#     'next':      ["If you want a crisp lock: /mod resonator → hold → tick 5", ...],
#     'lines':     ["• κ=0.82 — strong phase alignment (good).", "• p½=0.11 — far from the boundary (good).", ...]
#   }
#
# Notes:
#   • We *compute* κ, p½, residual/margins using elementfold.telemetry.measure.
#   • If folds (ℱ) are present we also report predicted retention A≈e^(−2ℱ).
#   • “z” is a simple *odds* proxy around the barrier: z = p½ / (1 − p½) (clamped).
#   • All fields are optional; we degrade gracefully and keep the REPL alive.
#
# MIT‑style tiny utility. © 2025 ElementFold authors.

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, List
import math
import torch

# Prefer absolute import for resilience inside the package.
from elementfold.telemetry import measure as telemetry_measure


# ──────────────────────────────────────────────────────────────────────────────
# Small helpers (robust and side‑effect free)
# ──────────────────────────────────────────────────────────────────────────────

def _as_tensor(x: Any, dtype=torch.float32) -> Optional[torch.Tensor]:
    if x is None:
        return None
    try:
        t = torch.as_tensor(x, dtype=dtype)
        return t
    except Exception:
        return None

def _finite(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default

def _safe_last_row_mean(t: Optional[torch.Tensor]) -> Optional[float]:
    if t is None:
        return None
    if t.numel() == 0:
        return None
    # Mean over batch, take last position along T (end‑of‑sequence).
    try:
        return float(t[:, -1].mean().item())
    except Exception:
        try:
            return float(t.flatten()[-1].item())
        except Exception:
            return None

def _classify(value: Optional[float], *, good_lo=None, good_hi=None, warn_lo=None, warn_hi=None) -> str:
    """
    Simple 3‑light classifier:
      • If value is None → 'unknown'
      • If a numeric band (good_lo..good_hi) is supplied and value in band → 'good'
      • Else if warn band supplied and value in band → 'warn'
      • Else → 'risk'
    """
    if value is None or not math.isfinite(value):
        return "unknown"
    if (good_lo is not None) and (good_hi is not None) and (good_lo <= value <= good_hi):
        return "good"
    if (warn_lo is not None) and (warn_hi is not None) and (warn_lo <= value <= warn_hi):
        return "warn"
    return "risk"

def _nearness_to_barrier(p_half: Optional[float]) -> Optional[float]:
    """0 = far from half‑step, 1 = right on it."""
    if p_half is None:
        return None
    d = abs(0.5 - float(p_half))
    return float(1.0 - min(max(d / 0.5, 0.0), 1.0))  # ∈ [0,1]

def _odds(p: Optional[float], cap: float = 10.0) -> Optional[float]:
    """z = p / (1 − p), clamped for comfort."""
    if p is None:
        return None
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return float(min(max(p / (1.0 - p), 0.0), cap))

def _retention_from_folds(F: Optional[float]) -> Optional[float]:
    """A ≈ e^(−2ℱ) — intuitive 'retention' proxy: 1 = unchanged, 0 → fully let go."""
    if F is None:
        return None
    return float(math.exp(-2.0 * float(F)))

def _fmt(x: Optional[float], nd: int = 3) -> str:
    if x is None or not math.isfinite(x):
        return "?"
    return f"{x:.{nd}f}"

def _combine_traffic(*labels: str) -> str:
    """
    Combine many lights into one conservative light,
    where risk > warn > good > unknown.
    """
    order = {"risk": 3, "warn": 2, "good": 1, "unknown": 0}
    s = max((order.get(l, 0) for l in labels), default=0)
    rev = {v: k for k, v in order.items()}
    return rev.get(s, "unknown")


# ──────────────────────────────────────────────────────────────────────────────
# Core: interpret()
# ──────────────────────────────────────────────────────────────────────────────

def interpret(
    X: Optional[torch.Tensor],
    *,
    delta: float,
    folds: Optional[torch.Tensor] = None,
    relax_meta: Optional[Dict[str, Any]] = None,
    control: Optional[Dict[str, float]] = None,
    rung: Optional[Dict[str, Any]] = None,
    tele_hint: Optional[Dict[str, Any]] = None,
    detail: bool = False,
) -> Dict[str, Any]:
    """
    Convert raw signals into a friendly explanation + suggestions.

    Returns a JSON‑friendly dict (see module header). Works with partial inputs.
    """
    # 0) Prepare inputs
    X = _as_tensor(X, torch.float32)
    F = _as_tensor(folds, torch.float32)

    # 1) Measure coherence on δ⋆ circle (robust to shapes / None)
    tele: Dict[str, Any]
    if X is not None:
        tele = telemetry_measure(X, float(delta), detail=detail)
    else:
        # Fall back to whatever hints we got (or empty defaults).
        tele = {
            "kappa": _finite((tele_hint or {}).get("kappa"), 0.0),
            "p_half": _finite((tele_hint or {}).get("p_half"), 0.0),
            "margin_mean": _finite((tele_hint or {}).get("margin_mean"), 0.0),
            "margin_min": _finite((tele_hint or {}).get("margin_min"), 0.0),
            "resid_std": _finite((tele_hint or {}).get("resid_std"), 0.0),
            "phase_mean": _finite((tele_hint or {}).get("phase_mean"), 0.0),
            "delta": float(delta),
        }

    kappa = float(tele.get("kappa", 0.0))
    p_half = float(tele.get("p_half", 0.0))
    margin_min = float(tele.get("margin_min", 0.0))
    resid_std = float(tele.get("resid_std", 0.0))
    phase_mean = float(tele.get("phase_mean", 0.0))
    delta_used = float(tele.get("delta", delta))

    # 2) Derived proxies
    near = _nearness_to_barrier(p_half)               # 0..1
    z = _odds(p_half)                                 # odds near barrier
    F_last = _safe_last_row_mean(F)                   # scalar or None
    A = _retention_from_folds(F_last)                 # predicted retention

    # 3) Traffic lights (simple but sensible defaults)
    #    • Coherence (κ): good ≥0.75, warn ≥0.45 else risk.
    #    • Barrier (nearness): good ≤0.25, warn ≤0.55, risk otherwise.
    #    • Stability from residual spread (vs. half‑delta).
    #    • Pace from mean Δℱ (if folds present).
    coh_light = _classify(kappa, good_lo=0.75, good_hi=1.0, warn_lo=0.45, warn_hi=0.75)
    bar_light = _classify(near, good_lo=0.0, good_hi=0.25, warn_lo=0.25, warn_hi=0.55)

    half = delta_used * 0.5
    resid_norm = resid_std / max(half, 1e-9) if math.isfinite(half) else float("inf")
    stab_light = _classify(resid_norm, good_lo=0.0, good_hi=0.20, warn_lo=0.20, warn_hi=0.40)

    pace_light = "unknown"
    if F is not None and F.numel() > 1:
        dF = F[:, 1:] - F[:, :-1]
        pace = float(dF.mean().item())
        pace_light = _classify(pace, good_lo=0.0, good_hi=0.05, warn_lo=0.05, warn_hi=0.15)

    # 4) Rung FSM state (if provided)
    intent = str((rung or {}).get("intent", "stabilize"))
    phase = str((rung or {}).get("phase", "?"))
    dwell = int((rung or {}).get("dwell", 0))
    band = float((rung or {}).get("band", delta_used / 6.0))

    # 5) Control knobs (if provided)
    beta = _finite((control or {}).get("beta"), float("nan"))
    gamma = _finite((control or {}).get("gamma"), float("nan"))
    clamp = _finite((control or {}).get("clamp"), float("nan"))

    # 6) Relaxation meta (safe echoes)
    relax = dict(relax_meta or {})
    eta = _finite(relax.get("eta"), 0.0)
    lam = _finite(relax.get("lambda"), 0.0)
    D   = _finite(relax.get("D"), 0.0)
    rho = _finite(relax.get("rho"), 0.0)
    dt  = _finite(relax.get("dt"), 1.0)
    steps = int(relax.get("steps", 1))

    # 7) Build sentences (non‑expert, terse + precise)
    lines: List[str] = []
    # Coherence
    if coh_light == "good":
        lines.append(f"• κ={_fmt(kappa)} — strong phase alignment (good).")
    elif coh_light == "warn":
        lines.append(f"• κ={_fmt(kappa)} — moderate alignment; consider HOLD to center.")
    else:
        lines.append(f"• κ={_fmt(kappa)} — weak alignment (risk); use HOLD or lower β / raise γ to settle.")

    # Barrier
    if near is not None:
        if bar_light == "good":
            lines.append(f"• p½={_fmt(p_half)} — far from the boundary (good).")
        elif bar_light == "warn":
            lines.append(f"• p½={_fmt(p_half)} — somewhat near the boundary; raise γ if chatter appears.")
        else:
            lines.append(f"• p½={_fmt(p_half)} — at/near the half‑step (risk); step or damp before continuing.")
    else:
        lines.append("• p½=? — barrier proximity unknown (no ledger).")

    # Residuals / margins
    if math.isfinite(margin_min):
        if margin_min < 0.0:
            lines.append(f"• margin_min={_fmt(margin_min)} — already over the line; capture a rung.")
        else:
            lines.append(f"• margin_min={_fmt(margin_min)} — safety gap to boundary (higher is safer).")

    if math.isfinite(resid_std):
        lines.append(f"• resid_std={_fmt(resid_std)} — residual spread within a click (lower is calmer).")

    # Relaxation
    if F_last is not None:
        lines.append(f"• ℱ={_fmt(F_last)} — accumulation of sharing/steps so far.")
        if A is not None:
            lines.append(f"• A≈e^(−2ℱ)={_fmt(A)} — predicted retention; smaller A → stronger letting‑go.")
    if any(v > 0 for v in (eta, lam, D, rho)):
        lines.append(
            "• relax: "
            + ", ".join([
                f"η={_fmt(eta,3)}",
                f"λ={_fmt(lam,3)}",
                f"D={_fmt(D,3)}",
                f"ρ={_fmt(rho,3)}",
                f"dt={_fmt(dt,3)}×{steps}",
            ])
            + " — diffusion/decay is active."
        )

    # Control summary
    if math.isfinite(beta) or math.isfinite(gamma) or math.isfinite(clamp):
        btxt = f"β={_fmt(beta,2)}" if math.isfinite(beta) else "β=?"
        gtxt = f"γ={_fmt(gamma,2)}" if math.isfinite(gamma) else "γ=?"
        ctxt = f"⛔={_fmt(clamp,1)}" if math.isfinite(clamp) else "⛔=?"
        lines.append(f"• control: {btxt}  {gtxt}  {ctxt}.")

    # FSM and acceptance band
    lines.append(f"• rung: intent={intent}, phase={phase}, dwell={dwell}, band≈{_fmt(band)}.")

    # 8) Suggestions (context‑aware; short, concrete)
    next_actions: List[str] = []
    # Coherence guidance
    if coh_light == "risk":
        next_actions.append("Use HOLD to recenter; if still noisy, increase γ slightly (e.g., +0.1).")
    elif coh_light == "warn":
        next_actions.append("If response feels loose, try a short HOLD, then SEEK one rung.")
    # Barrier guidance
    if bar_light == "risk":
        next_actions.append("You are on the ridge: either step up/down or raise γ to avoid teetering.")
    # Pace
    if pace_light == "warn":
        next_actions.append("Relaxation is ramping: consider lowering η or ρ to slow temperature lift.")
    # Margin guidance
    if margin_min < 0.0:
        next_actions.append("Margin < 0: complete capture before further steps (stay in CAPTURE until κ increases).")
    # Generic rung usage
    if intent.lower() != "hold" and bar_light in {"warn", "risk"}:
        next_actions.append("Plan: HOLD → tick 3, then SEEK toward target rung when κ stabilizes.")

    # 9) Caption + headline (for Studio / UI badges)
    badges = {
        "F": (None if F_last is None else float(F_last)),
        "z": (None if z is None else float(z)),
        "A": (None if A is None else float(A)),
    }

    # Friendly caption tying control + quick qualitative state
    qual = []
    if coh_light == "good":
        qual.append("coherent")
    elif coh_light == "warn":
        qual.append("moderate")
    else:
        qual.append("noisy")

    if bar_light == "good":
        qual.append("far from barrier")
    elif bar_light == "warn":
        qual.append("near barrier")
    else:
        qual.append("on barrier")

    caption = (
        f"β={_fmt(beta,2)}  γ={_fmt(gamma,2)}  ⛔={_fmt(clamp,1)}"
        f"  |  {', '.join(qual)}"
    )

    # Headline with badges the UI can parse (ℱ / z / A tokens are deliberate)
    intent_tag = phase if phase != "?" else intent.upper()
    parts = [intent_tag]
    parts.append(f"κ={_fmt(kappa,2)}")
    parts.append(f"p½={_fmt(p_half,2)}")
    if F_last is not None:
        parts.append(f"ℱ={_fmt(F_last,3)}")
    if z is not None:
        parts.append(f"z={_fmt(z,2)}")
    if A is not None:
        parts.append(f"A≈e^(−2ℱ)={_fmt(A,2)}")
    headline = " • ".join(parts)

    # 10) Metrics table (each with value + plain‑English meaning)
    metrics: Dict[str, Dict[str, Any]] = {
        "kappa": {
            "value": float(kappa),
            "meaning": "phase alignment on δ⋆ circle (1 = tight, 0 = spread)",
            "light": coh_light,
            "range_hint": "[0..1] higher is better",
        },
        "p_half": {
            "value": float(p_half),
            "meaning": "fraction near the half‑step boundary (0 = safe center, 0.5 = ridge)",
            "light": bar_light,
            "range_hint": "[0..1] lower is safer",
        },
        "margin_min": {
            "value": float(margin_min),
            "meaning": "closest distance to boundary in X (negative means already over)",
            "light": ("risk" if margin_min < 0 else "good"),
            "range_hint": "≥ 0 preferred",
        },
        "resid_std": {
            "value": float(resid_std),
            "meaning": "spread of residuals inside a click (smaller is calmer)",
            "light": stab_light,
            "range_hint": f"~0..{_fmt(0.5*delta_used)} (relative to δ⋆/2)",
        },
        "phase_mean": {
            "value": float(phase_mean),
            "meaning": "average location in X modulo δ⋆",
            "light": "unknown",
            "range_hint": f"[0..{_fmt(delta_used)}), informational",
        },
        "delta": {
            "value": float(delta_used),
            "meaning": "click size δ⋆",
            "light": "unknown",
            "range_hint": "model/driver setting",
        },
        "F_last": {
            "value": (None if F_last is None else float(F_last)),
            "meaning": "cumulative fold counter at sequence end",
            "light": pace_light if F_last is not None else "unknown",
            "range_hint": "grows with distance/effort",
        },
        "A_retention": {
            "value": (None if A is None else float(A)),
            "meaning": "predicted retention A≈e^(−2ℱ) (1=unchanged, 0=let go)",
            "light": "unknown" if A is None else ("good" if A >= 0.6 else ("warn" if A >= 0.3 else "risk")),
            "range_hint": "[0..1]",
        },
        "odds_z": {
            "value": (None if z is None else float(z)),
            "meaning": "odds proxy near barrier: p½/(1−p½)",
            "light": bar_light,
            "range_hint": "≈1 at ridge, ↓ when centered",
        },
    }

    # 11) Traffic roll‑up (conservative)
    traffic = {
        "coherence": coh_light,
        "barrier": bar_light,
        "stability": stab_light,
        "pace": pace_light,
        "overall": _combine_traffic(coh_light, bar_light, stab_light, pace_light),
    }

    return {
        "headline": headline,
        "caption": caption,
        "badges": badges,
        "traffic": traffic,
        "metrics": metrics,
        "next": next_actions,
        "lines": lines,
        # Echoes (handy for UIs that want raw state)
        "rung": {"intent": intent, "phase": phase, "dwell": dwell, "band": band},
        "relax": {"eta": eta, "lambda": lam, "D": D, "rho": rho, "dt": dt, "steps": steps},
    }


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: compact status lines for Studio (optional)
# ──────────────────────────────────────────────────────────────────────────────

def format_status_block(state: Dict[str, Any]) -> str:
    """
    Render a compact status block from interpret(...) output.
    Safe for REPL printing; no colors (Studio prints its own gauges).
    """
    h = state.get("headline", "")
    cap = state.get("caption", "")
    traffic = state.get("traffic", {})
    overall = traffic.get("overall", "unknown")
    nxt = state.get("next", [])[:2]  # keep short
    nx = (" • ".join(nxt)) if nxt else ""
    tail = f"\n↳ next: {nx}" if nx else ""
    return f"{h}\n{cap}\ntraffic={overall}{tail}"
