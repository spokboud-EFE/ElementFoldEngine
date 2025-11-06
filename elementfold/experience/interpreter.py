# ElementFold · experience/interpreter.py
# ──────────────────────────────────────────────────────────────────────────────
# TelemetryInterpreter — anchors live tensors/telemetry to a *short, precise*
# human narrative + next-step suggestions. It uses a local LLM when available,
# and a deterministic rule set as a safe fallback.
#
# Inputs (typical keys you can pass in `telemetry`):
#   {'kappa'|'κ', 'p_half'|'p½', 'margin_min', 'margin_mean', 'resid_std',
#    'phase_mean', 'delta'|'δ⋆', 'folds' (B×T or T), 'relax_meta': {...}}
#
# Optional controller `status` (adapter-supplied):
#   {'intent','k_target'| 'target_k','phase','band','dwell','plan'}
#
# Output (JSON-friendly dict):
#   {
#     'caption': "✓ κ=0.91 — captured and safe (p½=0.05)",
#     'bullets': ["phase alignment strong", "safe margin from boundary", ...],
#     'next_actions': ["tick 2","hold"],
#     'numbers': {... scalar snapshot ...},
#     'confidence': 0.0..1.0,
#     'llm_used': True|False,
#     'raw_llm': "<raw text>",            # optional, for debugging
#     'llm_meta': {'backend':..., 'latency_ms':...}  # optional
#   }
#
# Design goals:
#   • Always say something useful (deterministic fallback is precise & safe).
#   • When an LLM is available, keep it *anchored to numbers* and ask for
#     concise, actionable coaching (no fluff, no risky moves near boundaries).
#   • No heavy imports at file-top; optional LLM wrapper is lazy/guarded.
#
# © 2025 ElementFold authors. MIT-ish tiny utility.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math
import re

# Optional LLM bridge (kept lazy and guarded). This file compiles without it.
try:  # pragma: no cover
    from .background_model import BackgroundModel, PilotConfig, as_messages  # type: ignore
except Exception:  # pragma: no cover
    BackgroundModel = None     # type: ignore
    PilotConfig = None         # type: ignore

    def as_messages(system: str, user: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

Number = float
DictStrAny = Dict[str, Any]


# ──────────────────────────────────────────────────────────────────────────────
# Small helpers (sanitization, stats)
# ──────────────────────────────────────────────────────────────────────────────

def _get_float(d: DictStrAny, *keys: str, default: Optional[float] = None) -> Optional[float]:
    """Return the first finite float found among candidate keys, else default."""
    for k in keys:
        if k in d:
            try:
                v = float(d[k])
                if math.isfinite(v):
                    return v
            except Exception:
                pass
    return default


def _first_row_stats(x: Any) -> Dict[str, float]:
    """
    Accept a tensor-like or list-like 'folds' (B,T) or (T,) and return simple stats
    on the first row: {'mean','max','last'}. Returns {} if parsing fails.
    """
    # Torch path (optional)
    try:
        import torch  # local import; may fail in stripped environments
        if isinstance(x, torch.Tensor):
            t = x.detach().cpu()
            if t.dim() == 2:
                t = t[0]
            elif t.dim() != 1:
                return {}
            if t.numel() == 0:
                return {}
            return {
                "mean": float(t.mean().item()),
                "max": float(t.max().item()),
                "last": float(t[-1].item()),
            }
    except Exception:
        pass

    # Sequence path
    try:
        seq = list(x[0]) if (x and isinstance(x[0], (list, tuple))) else list(x)
        if not seq:
            return {}
        n = len(seq)
        m = sum(float(v) for v in seq) / max(1, n)
        mx = max(float(v) for v in seq)
        return {"mean": float(m), "max": float(mx), "last": float(seq[-1])}
    except Exception:
        return {}


def _grade_band(kappa: Optional[float], p_half: Optional[float], margin_min: Optional[float]) -> str:
    """
    Coarse UX “color” band (textual):
      • "green"   — captured & safe
      • "yellow"  — cautious
      • "red"     — risky (near/over boundary)
    """
    if kappa is None:
        return "yellow"
    if margin_min is not None and margin_min < 0.0:
        return "red"
    if p_half is not None and p_half > 0.30:
        return "red"
    if kappa > 0.85:
        return "green"
    if kappa > 0.65:
        return "yellow"
    return "red"


def _suggest_commands(
    kappa: Optional[float],
    p_half: Optional[float],
    margin_min: Optional[float],
    status: Optional[DictStrAny],
) -> List[str]:
    """
    Conservative adapter text commands for REPL/adapters.
    """
    cmds: List[str] = []
    phase = (status or {}).get("phase")

    # Safety first
    if margin_min is not None and margin_min < 0.0:
        return ["hold", "tick 5"]

    # Near boundary — stabilize gently
    if p_half is not None and p_half > 0.30:
        return ["hold", "tick 3"]

    # Weak capture → probe slowly
    if kappa is not None and kappa < 0.55:
        return ["tick 3", "step up 1"]

    # Well captured → short maintenance
    if kappa is not None and kappa >= 0.85:
        base = ["tick 2"]
        if phase and str(phase).upper() != "CAPTURED":
            base.append("hold")
        return base

    # Default
    return ["tick 2"]


def _deterministic_narrative(tele: DictStrAny, status: Optional[DictStrAny]) -> Tuple[str, List[str], float]:
    """
    Rules-only caption + bullets + confidence (0..1).
    """
    kappa = _get_float(tele, "kappa", "κ", default=None)
    p_half = _get_float(tele, "p_half", "p½", default=None)
    margin_min = _get_float(tele, "margin_min", default=None)
    margin_mean = _get_float(tele, "margin_mean", default=None)
    resid_std = _get_float(tele, "resid_std", default=None)
    delta = _get_float(tele, "delta", "δ⋆", default=None)

    band = _grade_band(kappa, p_half, margin_min)

    # Caption (one line, quick signal)
    if isinstance(kappa, float):
        marker = "✓" if band == "green" else ("!" if band == "red" else "•")
        caption = f"{marker} κ={kappa:.3f}"
    else:
        caption = "• assessing coherence"

    # Bullets (one fact per line)
    bullets: List[str] = []
    if kappa is not None:
        if kappa >= 0.85:
            bullets.append("phase alignment strong (captured)")
        elif kappa >= 0.65:
            bullets.append("phase moderately aligned (watch drift)")
        else:
            bullets.append("phase weak — consider probing or holding")

    if p_half is not None:
        if p_half > 0.30:
            bullets.append(f"near half‑click boundary (p½={p_half:.2f}) — risk of rung flip")
        elif p_half > 0.10:
            bullets.append(f"some proximity to boundary (p½={p_half:.2f})")
        else:
            bullets.append("safe margin from boundary")

    if margin_min is not None:
        if margin_min < 0.0:
            bullets.append(f"margin_min={margin_min:.4f} — already over the line; hold & damp")
        elif margin_mean is not None:
            bullets.append(f"safety gap ~{margin_mean:.4f} (min {margin_min:.4f})")

    if resid_std is not None:
        bullets.append(f"residual spread σ≈{resid_std:.4f} (narrower is calmer)")

    if delta is not None:
        bullets.append(f"δ⋆={delta:g} (click size)")

    if status:
        ph = status.get("phase")
        it = status.get("intent")
        if ph:
            bullets.append(f"controller phase: {ph}")
        if it:
            bullets.append(f"controller intent: {it}")

    # Confidence heuristic: rise with kappa, shrink with boundary risk
    conf = 0.5
    if isinstance(kappa, float):
        conf = 0.4 + 0.6 * max(0.0, min(1.0, kappa))
    if isinstance(p_half, float):
        conf *= max(0.2, 1.0 - 0.7 * max(0.0, min(1.0, p_half)))
    if isinstance(margin_min, float) and margin_min < 0.0:
        conf *= 0.6

    return caption, bullets, float(max(0.05, min(0.98, conf)))


# ──────────────────────────────────────────────────────────────────────────────
# Interpreter
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class InterpreterConfig:
    enable_llm: bool = True
    max_new_tokens: int = 160
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 0
    system_prompt: str = (
        "You are ElementFold Co‑Pilot. You see live telemetry on a δ⋆ rung ladder. "
        "Explain numbers precisely, avoid jargon, and propose next adapter commands "
        "like ['hold','tick 3','step up 1'] when safe. ≤6 bullets; be conservative near boundaries."
    )


class TelemetryInterpreter:
    """
    Bridge from raw telemetry → human narrative (+ suggested commands).
    Uses a BackgroundModel (LLM) when healthy; falls back to deterministic rules.
    """
    def __init__(self, pilot: Any = None, cfg: Optional[InterpreterConfig] = None):
        self.cfg = cfg or InterpreterConfig()
        self._pilot = pilot  # may be None (then we always fall back)

    def summarize(
        self,
        telemetry: DictStrAny,
        *,
        status: Optional[DictStrAny] = None,
        context: Optional[DictStrAny] = None,
    ) -> DictStrAny:
        # 1) Extract cleaned numbers
        kappa = _get_float(telemetry, "kappa", "κ")
        p_half = _get_float(telemetry, "p_half", "p½")
        margin_min = _get_float(telemetry, "margin_min")
        margin_mean = _get_float(telemetry, "margin_mean")
        resid_std = _get_float(telemetry, "resid_std")
        delta = _get_float(telemetry, "delta", "δ⋆")
        folds_stats = _first_row_stats(telemetry.get("folds")) if telemetry.get("folds", None) is not None else {}

        # 2) Deterministic baseline
        cap0, bullets0, conf0 = _deterministic_narrative(telemetry, status)
        next0 = _suggest_commands(kappa, p_half, margin_min, status)

        # 3) If LLM disabled or not ready, return baseline
        pilot_ready = bool(self._pilot and hasattr(self._pilot, "ready") and self._pilot.ready())
        if (not self.cfg.enable_llm) or (not pilot_ready):
            return {
                "caption": cap0,
                "bullets": bullets0,
                "next_actions": next0,
                "numbers": {
                    "kappa": kappa, "p_half": p_half, "margin_min": margin_min, "margin_mean": margin_mean,
                    "resid_std": resid_std, "delta": delta, "folds_stats": folds_stats or None
                },
                "confidence": conf0,
                "llm_used": False,
            }

        # 4) Build an anchored chat prompt for the LLM (facts first)
        lines = []
        def add(k: str, v: Optional[float]):
            if isinstance(v, float):
                lines.append(f"{k}={v:.6g}")

        add("kappa", kappa); add("p_half", p_half); add("margin_min", margin_min); add("margin_mean", margin_mean)
        add("resid_std", resid_std); add("delta", delta)
        if folds_stats:
            add("folds.mean", folds_stats.get("mean"))
            add("folds.max", folds_stats.get("max"))
            add("folds.last", folds_stats.get("last"))

        if status:
            ph = status.get("phase"); it = status.get("intent"); kt = status.get("k_target", status.get("target_k"))
            if ph: lines.append(f"controller.phase={ph}")
            if it: lines.append(f"controller.intent={it}")
            if kt is not None: lines.append(f"controller.k_target={kt}")

        if context:
            for k, v in context.items():
                try:
                    lines.append(f"{k}={v}")
                except Exception:
                    pass

        facts_block = "\n".join(lines) if lines else "(no metrics)"
        baseline_block = "\n".join(["# baseline.caption", cap0, "# baseline.actions", *next0])

        user_prompt = f"""\
Here are live metrics (one per line):
{facts_block}

Provide:
1) A one-line status starting with ✓ / • / ! and include kappa/p_half if known.
2) 3–6 short bullets, each interpreting exactly one numeric fact (no fluff).
3) A JSON array of 2–4 *safe* next text commands for the adapter (e.g., ["hold","tick 3"]).

Constraints:
- Be conservative near boundaries (p_half>0.3 or margin_min<0).
- Prefer 'hold' + short 'tick N' for stabilization; only suggest 'step up|down' when clearly safe.
- Keep technical terms parenthesized; make it human-readable.
- If uncertain, say 'uncertain' and fall back to baseline.

Baseline (for reference):
{baseline_block}
"""

        # 5) Query the LLM
        try:
            msgs = as_messages(system=self.cfg.system_prompt, user=user_prompt)
            comp = self._pilot.generate(  # type: ignore[attr-defined]
                msgs,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                top_k=self.cfg.top_k,
            )
            raw = (comp.text or "").strip()
            suggested = _extract_json_commands(raw) or next0
            caption = _extract_first_line(raw) or cap0
            bullets = _extract_bullets(raw) or bullets0

            return {
                "caption": caption,
                "bullets": bullets,
                "next_actions": suggested[:4],
                "numbers": {
                    "kappa": kappa, "p_half": p_half, "margin_min": margin_min, "margin_mean": margin_mean,
                    "resid_std": resid_std, "delta": delta, "folds_stats": folds_stats or None
                },
                "confidence": max(conf0, 0.7),
                "llm_used": True,
                "raw_llm": raw,
                "llm_meta": {
                    "backend": getattr(comp, "backend", None),
                    "latency_ms": getattr(comp, "latency_ms", None),
                },
            }
        except Exception:
            # Graceful fallback if anything goes wrong upstream
            return {
                "caption": cap0,
                "bullets": bullets0,
                "next_actions": next0,
                "numbers": {
                    "kappa": kappa, "p_half": p_half, "margin_min": margin_min, "margin_mean": margin_mean,
                    "resid_std": resid_std, "delta": delta, "folds_stats": folds_stats or None
                },
                "confidence": conf0,
                "llm_used": False,
            }


# ──────────────────────────────────────────────────────────────────────────────
# Tiny, robust extractors (tolerant to loosely formatted LLM output)
# ──────────────────────────────────────────────────────────────────────────────

def _extract_first_line(text: str) -> Optional[str]:
    for line in text.splitlines():
        s = line.strip()
        if s:
            return s
    return None


def _extract_bullets(text: str) -> List[str]:
    """
    Collect up to 6 bullet-ish lines. Accepts '-', '•', or '1.'/'1)' markers.
    """
    bullets: List[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("-") or s.startswith("•") or re.match(r"^\d+[\.\)]\s+", s):
            s = re.sub(r"^(\-|\•|\d+[\.\)])\s*", "", s).strip()
            if s:
                bullets.append(s)
        if len(bullets) >= 6:
            break
    return bullets


def _extract_json_commands(text: str) -> Optional[List[str]]:
    """
    Find the first JSON array of strings in the text (e.g., ["hold","tick 3"]).
    """
    try:
        import json
        m = re.search(r"\[[^\]]+\]", text, flags=re.S)
        if not m:
            return None
        arr = json.loads(m.group(0))
        if isinstance(arr, list) and all(isinstance(x, str) for x in arr):
            out = [s.strip().lower() for s in arr if isinstance(s, str) and s.strip()]
            return out or None
        return None
    except Exception:
        return None
