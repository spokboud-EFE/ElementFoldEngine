# ElementFold · experience/steering.py
# ──────────────────────────────────────────────────────────────────────────────
# The SteeringController turns *intent text* into a compact control vector:
#   v ∈ ℝ⁸ = [β, γ, ⛔, style₅]
#
# Meanings:
#   • β (beta)    — gate exposure (how strongly FGN exposes novelty),
#   • γ (gamma)   — normalization damping (how hard FGN calms energy),
#   • ⛔ (clamp)  — gate cap (how deep negative gate values can go before clipping),
#   • style₅      — five free “style” scalars adapters can interpret (tone, tempo, etc.).
#
# Design goals:
#   • Minimal & fast: tokenizer → ids → embedding → mean‑pool → 2‑layer MLP → ℝ⁸.
#   • Trainable: see steering_train.py; defaults work out‑of‑the‑box.
#   • Safe ranges: a helper maps raw outputs into Supervisor‑aligned bounds.
#   • NEW: Pilot mode — online reliability‑aware nudger + light head‑only learning.
#
# Contract with Studio:
#   ctrl = SteeringController.load_default(cfg.delta)
#   v    = ctrl("gentle, coherent")               # → ℝ⁸
#   p    = SteeringController.to_params(v)        # → {'beta','gamma','clamp','style','style_norm'}
#
from __future__ import annotations

from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass, field
import math
import torch
import torch.nn as nn

from ..core.tokenizer import SimpleTokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sigmoid_range(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Map ℝ → (lo, hi) smoothly via σ. Works element‑wise and is differentiable."""
    return torch.sigmoid(x) * (hi - lo) + lo

def _inv_sigmoid_range(y: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """
    Inverse of _sigmoid_range for scalar y in (lo, hi):
        x = logit((y - lo) / (hi - lo))
    Clamps y slightly inside (lo,hi) to avoid ±∞.
    """
    eps = 1e-4
    p = torch.clamp((y - lo) / (hi - lo + 1e-12), eps, 1.0 - eps)
    return torch.log(p) - torch.log1p(-p)

def _tanh_range(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Map ℝ → [lo, hi] with tanh; handy for style normalization into [-1,1] or a narrow band."""
    mid = 0.5 * (hi + lo)
    half = 0.5 * (hi - lo)
    return mid + half * torch.tanh(x)

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _frac_step_range(curr: float, lo: float, hi: float, frac: float, sign: float) -> float:
    """Take a bounded fractional step of the control's available range."""
    width = max(1e-8, hi - lo)
    delta = width * abs(frac) * (1.0 if sign >= 0 else -1.0)
    return _clamp(curr + delta, lo, hi)

def _safe_float(x: Any, default: float) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


# ──────────────────────────────────────────────────────────────────────────────
# Pilot configuration/state
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PilotConfig:
    """
    Reliability‑aware pilot that can gently nudge (β, γ, ⛔) and optionally learn online.
    """
    enabled: bool = False
    reliability_threshold: float = 0.78   # takeover threshold
    dwell_steps: int = 12                 # consecutive good steps before takeover
    max_frac_step: float = 0.08           # ≤ 8% of range per nudge
    lr: float = 1e-3                      # head‑only learning rate
    online_learn: bool = True             # adapt the head to your usage
    ewma_alpha: float = 0.20              # smoothing for telemetry signals
    rails: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "beta": (0.5, 2.0),
        "gamma": (0.0, 0.9),
        "clamp": (1.0, 10.0),
    })
    # Feature weights → reliability
    w_quality: float = 0.45
    w_stability: float = 0.35
    w_novelty: float = 0.20
    # Safety rails if telemetry screams (hard stops, still inside rails)
    gamma_hard_hi: float = 0.98
    beta_hard_hi: float = 2.20
    clamp_hard_hi: float = 12.0

@dataclass
class PilotState:
    ewma_quality: float = 0.5
    ewma_stability: float = 0.5
    ewma_novelty: float = 0.5
    reliability: float = 0.0
    consecutive_good: int = 0
    steps: int = 0
    last_params: Dict[str, float] = field(default_factory=dict)
    last_intent: str = ""
    last_reason: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# SteeringController — core + pilot
# ──────────────────────────────────────────────────────────────────────────────

class SteeringController(nn.Module):  # 🎚 intent → (β, γ, ⛔, style₅)
    """
    A tiny intent→control head. Forward returns a raw ℝ⁸ vector; use .to_params()
    to map into meaningful ranges.

    Notes:
      • δ⋆ (delta) is cached for convenience—some adapters may want to read it.
      • Tokenizer is intentionally tiny (vocab≈256); mean‑pooling is robust for short prompts.
      • Ranges can match Supervisor/RungController rails by passing `rails=...` to to_params().
      • NEW: enable_pilot(...) adds a reliability‑aware pilot that can take over.
    """

    # For very long prompts, we cap length to keep latency predictable.
    MAX_TOKENS: int = 512

    def __init__(self, delta: float = 0.030908106561043047, vocab: int = 256):
        super().__init__()
        self.delta = float(delta)               # δ⋆ cached (read‑only convenience)

        # — Embedding —
        # Keep in sync with SimpleTokenizer (vocab size ~ 256). Each token → ℝ⁶⁴.
        self.emb = nn.Embedding(vocab, 64)

        # — Head (MLP) —
        # A tiny two‑layer perceptron: ℝ⁶⁴ → ℝ⁸ = [β̂, γ̂, ⛔̂, style₅].
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
        )

        # Cache a tokenizer instance to avoid per‑call construction overhead.
        self._tok = SimpleTokenizer(vocab=vocab) if hasattr(SimpleTokenizer, "__call__") else SimpleTokenizer()

        # Pilot (disabled by default)
        self._pilot_cfg: Optional[PilotConfig] = None
        self._pilot_state: Optional[PilotState] = None
        self._pilot_opt: Optional[torch.optim.Optimizer] = None

    # ————————————————————————————————————————————————
    # Core forward (kept trainable for fine‑tuning)
    # ————————————————————————————————————————————————
    def forward(self, s: str) -> torch.Tensor:
        """
        Turn a text prompt into a raw control vector v ∈ ℝ⁸.

        Steps:
          1) tokenize prompt → ids,
          2) embed → (1, L, 64),
          3) mean‑pool across tokens → (1, 64),
          4) MLP → (1, 8),
          5) squeeze batch → (8,).

        Returns:
            torch.Tensor shape (8,), dtype float32 by default.
        """
        # Tokenize (allow empty input; cap length)
        try:
            ids = self._tok.encode(s or "")
        except Exception:
            ids = []
        if not ids:
            ids = [0]
        if len(ids) > self.MAX_TOKENS:
            ids = ids[: self.MAX_TOKENS]

        dev = self.emb.weight.device
        x = torch.as_tensor(ids, dtype=torch.long, device=dev).unsqueeze(0)  # (1, L)
        e = self.emb(x).mean(dim=1)                                          # (1, 64)
        v = self.fc(e).squeeze(0)                                            # (8,)
        return v

    # ————————————————————————————————————————————————
    # Helpers: map raw vector → meaningful ranges
    # ————————————————————————————————————————————————
    @staticmethod
    def to_params(
        v: torch.Tensor,
        rails: Optional[Dict[str, Tuple[float, float]]] = None,
        include_style_norm: bool = True,
    ) -> Dict[str, object]:
        """
        Convert a raw ℝ⁸ vector (as returned by forward) into interpretable parameters.

        Ranges:
            By default (conservative):
                β    ∈ [0.5, 2.0]
                γ    ∈ [0.0, 0.9]
                ⛔   ∈ [1.0, 10.0]
            To fully match RungController rails (see RungTuning), pass:
                rails={'beta':(0.5,3.0), 'gamma':(0.0,1.5), 'clamp':(1.0,12.0)}

        Returns:
            {
              'beta': float, 'gamma': float, 'clamp': float,
              'style': List[float],                 # raw style (length 5)
              'style_norm': List[float] (optional)  # tanh‑normalized to [-1,1] (length 5)
            }
        """
        # Rails (lo, hi) per control
        r = rails or {"beta": (0.5, 2.0), "gamma": (0.0, 0.9), "clamp": (1.0, 10.0)}

        with torch.no_grad():
            v = v.to(dtype=torch.float32)

            # Map first three controls into safe physical ranges via σ.
            beta  = float(_sigmoid_range(v[0], *r["beta"]))
            gamma = float(_sigmoid_range(v[1], *r["gamma"]))
            clamp = float(_sigmoid_range(v[2], *r["clamp"]))

            # Style block (unconstrained raw + normalized helper for UI sliders)
            style_vec = v[3:8].detach().cpu()
            style = [float(x) for x in style_vec.tolist()]
            out: Dict[str, object] = {"beta": beta, "gamma": gamma, "clamp": clamp, "style": style}

            if include_style_norm:
                # normalized to [-1,1] for immediate UI use
                s_norm = _tanh_range(style_vec, -1.0, 1.0).tolist()
                out["style_norm"] = [float(x) for x in s_norm]

        return out

    @staticmethod
    def describe(v: torch.Tensor, rails: Optional[Dict[str, Tuple[float, float]]] = None) -> str:
        """
        Human‑readable one‑liner for logs/Studio:
            "β=1.26  γ=0.43  ⛔=5.7  |  style≈[+0.31, −0.12, +0.04, +0.77, −0.05]"
        """
        p = SteeringController.to_params(v, rails=rails, include_style_norm=True)
        s = p.get("style_norm") or p.get("style") or []
        fmt = lambda x: f"{x:+.2f}"
        s_preview = ", ".join(fmt(x) for x in (s[:5] if isinstance(s, list) else []))
        return f"β={p['beta']:.2f}  γ={p['gamma']:.2f}  ⛔={p['clamp']:.1f}  |  style≈[{s_preview}]"

    # ————————————————————————————————————————————————
    # Convenience factories (untrained vs. checkpoint)
    # ————————————————————————————————————————————————
    @classmethod
    def load_default(cls, delta: float = 0.030908106561043047, vocab: int = 256) -> "SteeringController":
        """Create a fresh, untrained controller (useful for prototyping)."""
        return cls(delta, vocab=vocab)

    @classmethod
    def load(cls, path: str, delta: float = 0.030908106561043047, vocab: int = 256) -> "SteeringController":
        """Load weights from a state_dict checkpoint at `path`. Returns the controller in eval mode."""
        m = cls(delta, vocab=vocab)
        sd = torch.load(path, map_location="cpu")
        m.load_state_dict(sd)
        m.eval()
        return m

    # ————————————————————————————————————————————————
    # Optional: apply controls to a model directly
    # ————————————————————————————————————————————————
    def apply_to_model(self, model, s: str | None = None, v: torch.Tensor | None = None,
                       rails: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, object]:
        """
        Produce controls (from a prompt `s` or raw vector `v`) and push them
        into any model that implements `.apply_control(beta=?, gamma=?, clamp=?)`.

        Returns:
            The parameter dict actually applied (useful for logging/UX).
        """
        if v is None:
            if s is None:
                raise ValueError("either `s` (prompt) or `v` (raw ℝ⁸ vector) must be provided")
            v = self.forward(s)

        params = self.to_params(v, rails=rails or {"beta": (0.5, 2.0), "gamma": (0.0, 0.9), "clamp": (1.0, 10.0)})
        if hasattr(model, "apply_control"):
            model.apply_control(beta=params["beta"], gamma=params["gamma"], clamp=params["clamp"])
        return params

    # ─────────────────────────────────────────────────────────────────────────
    # NEW — Pilot mode: reliability score, takeover, gentle nudges, online learning
    # ─────────────────────────────────────────────────────────────────────────

    def enable_pilot(self, cfg: Optional[PilotConfig] = None) -> None:
        """
        Turn on the reliability‑aware pilot. Creates a tiny optimizer for the head (fc only).
        """
        self._pilot_cfg = cfg or PilotConfig(enabled=True)
        self._pilot_cfg.enabled = True
        self._pilot_state = PilotState()
        # Head‑only optimizer (keep emb stable for speed)
        self._pilot_opt = torch.optim.Adam(self.fc.parameters(), lr=self._pilot_cfg.lr)

    def pilot_enabled(self) -> bool:
        return bool(self._pilot_cfg and self._pilot_cfg.enabled)

    # ——— Telemetry helpers ———————————————————————————————————————

    @staticmethod
    def _novelty_from_text(text: Optional[str]) -> float:
        """
        Crude novelty proxy in [0,1]: unique trigram ratio (tokenized by spaces).
        Robust to missing text (returns 0.5).
        """
        if not isinstance(text, str) or not text.strip():
            return 0.5
        toks = text.split()
        if len(toks) < 6:
            return 0.6  # short outputs: bias slightly positive
        tri = [" ".join(toks[i:i+3]) for i in range(len(toks)-2)]
        uniq = len(set(tri))
        return _clamp(uniq / max(1, len(tri)), 0.0, 1.0)

    @staticmethod
    def _stability_from_phase_kappa(tele: Dict[str, Any]) -> float:
        """
        Map {phase, kap/kappa, p_half} → stability∈[0,1].
        Accepts multiple spellings from adapters or Studio.
        """
        phase = str(tele.get("phase") or tele.get("PHASE") or "").upper()
        kap = _safe_float(tele.get("kap") if "kap" in tele else tele.get("kappa"), float("nan"))
        p_half = _safe_float(tele.get("p_half") if "p_half" in tele else tele.get("p½"), float("nan"))

        base = 0.6
        if phase == "LOCKED":
            base = 0.85
        elif phase == "CAPTURE":
            base = 0.70
        elif phase == "MID":
            base = 0.55
        elif phase == "CROSSING":
            base = 0.35

        if not math.isnan(kap):
            base = 0.5 * base + 0.5 * _clamp(kap, 0.0, 1.0)
        if not math.isnan(p_half):
            # near barrier (≈0.5) is less stable; away from barrier is more stable
            away = 1.0 - 2.0 * abs(0.5 - _clamp(p_half, 0.0, 1.0))
            base = 0.7 * base + 0.3 * away

        return _clamp(base, 0.0, 1.0)

    @staticmethod
    def _quality_from_metrics(tele: Dict[str, Any]) -> float:
        """
        Map {loss, nll, ppl, entropy} → quality∈[0,1].
        Missing → 0.5 baseline.
        """
        if "loss" in tele:
            loss = _safe_float(tele["loss"], 2.0)
            return _clamp(1.0 / (1.0 + max(0.0, loss)), 0.0, 1.0)
        if "nll" in tele:
            nll = _safe_float(tele["nll"], 2.0)
            return _clamp(1.0 / (1.0 + max(0.0, nll)), 0.0, 1.0)
        if "ppl" in tele:
            ppl = _safe_float(tele["ppl"], 20.0)
            return _clamp(1.0 / (1.0 + (ppl / 10.0)), 0.0, 1.0)
        if "entropy" in tele:
            H = _safe_float(tele["entropy"], 4.0)  # 0..~10
            return _clamp(H / 10.0, 0.0, 1.0)
        return 0.5

    # ——— Pilot step ————————————————————————————————————————————————

    def pilot_step(
        self,
        intent_text: str,
        output_text: Optional[str],
        params_now: Dict[str, float],
        telemetry: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        One pilot tick. Computes reliability, optionally takes over and returns nudged params.

        Args:
            intent_text: the user intent line (what you typed in Studio).
            output_text: last model/adaptor textual output (for novelty proxy; may be None).
            params_now:  dict with current {'beta','gamma','clamp'} (floats).
            telemetry:   dict with optional keys, e.g.:
                         {'phase':'LOCKED'|'MID'|'CROSSING'|'CAPTURE',
                          'kap':0..1, 'p_half':0..1, 'loss':float|ppl:float,
                          'repeat_rate':0..1, 'clip_neg_rate':0..1, 'errors':int}

        Returns:
            {
              'reliability': float in [0,1],
              'takeover': bool,
              'params': {'beta','gamma','clamp'},   # updated if takeover else same as input
              'nudge': {'beta':Δ, 'gamma':Δ, 'clamp':Δ},  # proposed signed nudges (floats)
              'reason': str,                        # short human line
              'signals': {'quality','stability','novelty'}  # EWMA signals
            }
        """
        cfg = self._pilot_cfg or PilotConfig()  # safe defaults if not enabled
        st = self._pilot_state or PilotState()

        # 1) Signals → EWMA
        novelty = self._novelty_from_text(output_text)
        stability = self._stability_from_phase_kappa(telemetry)
        quality = self._quality_from_metrics(telemetry)

        a = float(cfg.ewma_alpha)
        st.ewma_novelty = (1 - a) * st.ewma_novelty + a * novelty
        st.ewma_stability = (1 - a) * st.ewma_stability + a * stability
        st.ewma_quality = (1 - a) * st.ewma_quality + a * quality

        # penalize if adapter reported errors
        if _safe_float(telemetry.get("errors", 0), 0.0) > 0.0:
            st.ewma_quality *= 0.85
            st.ewma_stability *= 0.85

        # 2) Reliability and dwell
        r = (cfg.w_quality * st.ewma_quality +
             cfg.w_stability * st.ewma_stability +
             cfg.w_novelty * st.ewma_novelty)
        st.reliability = _clamp(r, 0.0, 1.0)

        if st.reliability >= cfg.reliability_threshold:
            st.consecutive_good += 1
        else:
            st.consecutive_good = 0

        takeover = bool(self.pilot_enabled() and st.consecutive_good >= cfg.dwell_steps)

        # 3) Propose nudges (signed deltas), then apply if takeover
        beta, gamma, clamp = float(params_now["beta"]), float(params_now["gamma"]), float(params_now["clamp"])
        rb_lo, rb_hi = cfg.rails["beta"]
        rg_lo, rg_hi = cfg.rails["gamma"]
        rc_lo, rc_hi = cfg.rails["clamp"]

        nudge: Dict[str, float] = {"beta": 0.0, "gamma": 0.0, "clamp": 0.0}
        reasons: List[str] = []

        # Stabilize first: if stability weak, raise γ; if very weak, also soften β.
        if st.ewma_stability < 0.40:
            old = gamma
            gamma = _frac_step_range(gamma, rg_lo, rg_hi, cfg.max_frac_step * (1.5 if st.ewma_stability < 0.25 else 1.0), +1)
            nudge["gamma"] = gamma - old
            reasons.append("stability low → +γ")
            if st.ewma_stability < 0.30:
                old = beta
                beta = _frac_step_range(beta, rb_lo, rb_hi, cfg.max_frac_step * 0.5, -1)
                nudge["beta"] += (beta - old)
                reasons.append("very low stability → −β")

        # Novelty: if low but stability ok, increase β; if very high, tame β slightly.
        if st.ewma_novelty < 0.35 and st.ewma_stability >= 0.40:
            old = beta
            beta = _frac_step_range(beta, rb_lo, rb_hi, cfg.max_frac_step, +1)
            nudge["beta"] += (beta - old)
            reasons.append("novelty low → +β")
        elif st.ewma_novelty > 0.80 and st.ewma_quality < 0.60:
            old = beta
            beta = _frac_step_range(beta, rb_lo, rb_hi, cfg.max_frac_step * 0.5, -1)
            nudge["beta"] += (beta - old)
            reasons.append("novelty very high but quality low → −β")

        # Clamp: if many negatives are clipping, widen; if none and stable, narrow.
        clip_neg = _safe_float(telemetry.get("clip_neg_rate"), float("nan"))
        if not math.isnan(clip_neg):
            if clip_neg > 0.40:
                old = clamp
                clamp = _frac_step_range(clamp, rc_lo, rc_hi, cfg.max_frac_step, +1)
                nudge["clamp"] += (clamp - old)
                reasons.append("clip‑neg high → +⛔")
            elif clip_neg < 0.05 and st.ewma_stability > 0.7:
                old = clamp
                clamp = _frac_step_range(clamp, rc_lo, rc_hi, cfg.max_frac_step * 0.5, -1)
                nudge["clamp"] += (clamp - old)
                reasons.append("no clipping + stable → −⛔")

        # Hard safety stops (still inside declared rails)
        gamma = min(gamma, min(cfg.gamma_hard_hi, rg_hi))
        beta  = min(beta, min(cfg.beta_hard_hi, rb_hi))
        clamp = min(clamp, min(cfg.clamp_hard_hi, rc_hi))

        # If takeover: apply the nudges; else, keep params_now but report proposal.
        new_params = {"beta": beta, "gamma": gamma, "clamp": clamp} if takeover else dict(params_now)

        # 4) Online learning (head‑only): push controller outputs toward the chosen target
        if self.pilot_enabled() and cfg.online_learn:
            try:
                # Target is what we'd like the controller to emit next time for this intent.
                target = new_params if takeover else {
                    # even before takeover, bias the head toward proposed params (softly)
                    "beta": beta, "gamma": gamma, "clamp": clamp
                }
                self._pilot_train_step(intent_text, target, cfg.rails)
            except Exception:
                pass  # keep Studio alive no matter what

        st.steps += 1
        st.last_params = new_params
        st.last_intent = intent_text
        st.last_reason = " ; ".join(reasons) if reasons else "steady"

        return {
            "reliability": st.reliability,
            "takeover": takeover,
            "params": new_params,
            "nudge": nudge,
            "reason": st.last_reason,
            "signals": {
                "quality": st.ewma_quality,
                "stability": st.ewma_stability,
                "novelty": st.ewma_novelty,
            },
        }

    # ——— Head‑only online update —————————————————————————————

    def _pilot_train_step(self, intent_text: str, target_params: Dict[str, float],
                          rails: Dict[str, Tuple[float, float]]) -> None:
        """
        One tiny SGD step on the head (fc) so the controller learns
        to emit controls closer to `target_params` for this `intent_text`.

        We match *pre‑sigmoid* logits for the three controls to avoid saturation issues.
        """
        if self._pilot_opt is None:
            return

        self.train()
        dev = self.emb.weight.device

        # Forward
        v = self.forward(intent_text)  # (8,) on correct device

        # Compute target logits for β,γ,⛔
        with torch.no_grad():
            tb = torch.as_tensor([target_params["beta"]], device=dev, dtype=torch.float32)
            tg = torch.as_tensor([target_params["gamma"]], device=dev, dtype=torch.float32)
            tc = torch.as_tensor([target_params["clamp"]], device=dev, dtype=torch.float32)
            lb = _inv_sigmoid_range(tb, *rails["beta"])  # (1,)
            lg = _inv_sigmoid_range(tg, *rails["gamma"])
            lc = _inv_sigmoid_range(tc, *rails["clamp"])
            tgt_logits = torch.cat([lb, lg, lc], dim=0)  # (3,)

        pred_logits = v[:3]  # (3,)
        loss = torch.mean((pred_logits - tgt_logits) ** 2)

        self._pilot_opt.zero_grad(set_to_none=True)
        loss.backward()
        # Clip just in case
        torch.nn.utils.clip_grad_norm_(self.fc.parameters(), 1.0)
        self._pilot_opt.step()
        self.eval()
