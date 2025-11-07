# ElementFold · train.py
# ============================================================
# ElementFold training loop — “rung-centric” orchestration.
#
# The model learns while staying coherent on the δ⋆ circle:
#   • Supervisor keeps β (exposure), γ (damping), and ⛔ (clamp) in safe ranges.
#   • RungController handles discrete rung behavior (LOCK/HOLD/SEEK).
#   • Telemetry reports κ (coherence) and p½ (barrier proximity).
#
# This file glues everything together: model, optimizer, telemetry,
# control feedback, and friendly progress printing.
# ============================================================

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

# ──────────────────────────────────────────────────────────────────────────────
# Project imports with safe fallbacks (keep the loop runnable in minimal setups)
# ──────────────────────────────────────────────────────────────────────────────

from .model import Model
from .variational import VariationalLedger
from .telemetry import measure
from .control import Supervisor

# --- Align head (optional) ----------------------------------------------------
try:
    from .align import AlignHead  # expected: loss_align, pos, neg = align(xm, caps)
except Exception:  # pragma: no cover
    class AlignHead:  # type: ignore
        def __init__(self, delta: float) -> None:  # keep signature
            self.delta = float(delta)
        def to(self, device: torch.device) -> "AlignHead":
            return self
        def __call__(self, x_mean: torch.Tensor, caps: torch.Tensor):
            # No-op alignment: return zero loss, placeholders for pos/neg
            z = x_mean.new_zeros(())
            return z, None, None

# --- Optim (optional) ---------------------------------------------------------
try:
    from .optim import build_optimizer, make_scheduler, get_lr
except Exception:  # pragma: no cover
    def build_optimizer(model, lr: float = 2e-4, wd: float = 0.01):
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95), eps=1e-8)
    def make_scheduler(opt, warmup_steps: int, total_steps: int, min_lr_scale: float = 0.1):
        warmup_steps = max(1, int(warmup_steps))
        total_steps = max(warmup_steps + 1, int(total_steps))
        def lr_lambda(step: int):
            if step < warmup_steps:
                return float(step) / float(warmup_steps)
            # Cosine decay to min_lr_scale
            t = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
            cos = 0.5 * (1.0 + math.cos(math.pi * t))
            return min_lr_scale + (1.0 - min_lr_scale) * cos
        return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    def get_lr(opt) -> float:
        return float(opt.param_groups[0]["lr"])

# --- Data (optional) ----------------------------------------------------------
try:
    from .data import DataLoaderBuilder
except Exception:  # pragma: no cover
    class DataLoaderBuilder:  # type: ignore
        def __init__(self, *, seq_len: int, vocab: int, batch: int) -> None:
            self.seq_len, self.vocab, self.batch = int(seq_len), int(vocab), int(batch)
        def make(self):
            # Simple endless generator of random tokens (CPU; caller .to(device))
            class _DL:
                def __init__(self, L: int, V: int, B: int) -> None:
                    self.L, self.V, self.B = L, V, B
                def __iter__(self):
                    while True:
                        yield torch.randint(0, self.V, (self.B, self.L))
            return _DL(self.seq_len, self.vocab, self.batch)

# --- Rung controller (optional) ----------------------------------------------
try:
    from .rung_controller import RungController, RungIntent
except Exception:  # pragma: no cover
    class RungIntent:  # type: ignore
        STABILIZE = "stabilize"; HOLD = "hold"; SEEK = "seek"
        def __str__(self) -> str: return "stabilize"
    class RungController:  # type: ignore
        def __init__(self, *, delta: float, intent: Any, k_target: Optional[int], band: float) -> None:
            self.delta, self.intent, self.k_target, self.band = float(delta), intent, k_target, float(band)
        def update(self, telemetry: Dict[str, Any], ctrl_sup: Dict[str, float]) -> Dict[str, float]:
            # Pass-through: honor Supervisor output when the full controller is absent
            return dict(ctrl_sup)

# ──────────────────────────────────────────────────────────────────────────────
# UI helpers (Unicode-aware with ASCII fallback)
# ──────────────────────────────────────────────────────────────────────────────

def _supports_unicode() -> bool:
    enc = getattr(sys.stdout, "encoding", "") or ""
    return "UTF" in enc.upper()

def _glyphs(use_unicode: bool) -> Dict[str, str]:
    return ({
        "spin": "⟲", "ok": "✓", "warn": "⚠", "save": "💾",
        "beta": "β", "gamma": "γ", "clamp": "⛔", "delta": "δ⋆",
        "kappa": "κ", "phalf": "p½", "grad": "∥∇∥", "bolt": "⚡",
        "dot": "•", "bar_full": "▰", "bar_empty": "▱"
    } if use_unicode else {
        "spin": "*", "ok": "OK", "warn": "!", "save": "SAVE",
        "beta": "beta", "gamma": "gamma", "clamp": "CLAMP", "delta": "delta*",
        "kappa": "kappa", "phalf": "p_half", "grad": "||grad||", "bolt": ">",
        "dot": "-", "bar_full": "#", "bar_empty": "-"
    })

def _bar(frac: float, width: int, g: Dict[str, str]) -> str:
    frac = max(0.0, min(1.0, float(frac)))
    full = int(round(frac * width))
    return "[" + (g["bar_full"] * full) + (g["bar_empty"] * (width - full)) + f"] {int(frac*100):3d}%"

def _fmt(x: float | int | None, digits: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, int):
        return f"{x}"
    try:
        if abs(x) >= 1e3 or (0 < abs(x) < 1e-3):
            return f"{x:.2e}"
        return f"{x:.{digits}f}"
    except Exception:
        return str(x)

# ──────────────────────────────────────────────────────────────────────────────
# Rung helpers
# ──────────────────────────────────────────────────────────────────────────────

def _rung_metrics(x: torch.Tensor, delta: float) -> Dict[str, torch.Tensor]:
    """Compute rung index and residual for a batch of anchored-log values."""
    d = float(delta)
    k = torch.round(x / d)
    r = x - k * d
    return {"k": k, "r": r, "r_clicks": r / d}

def _rung_penalty(x_means: torch.Tensor, delta: float, band: float, intent: str) -> torch.Tensor:
    """Optional gentle penalty encouraging/avoiding proximity to rungs."""
    d = float(delta)
    k = torch.round(x_means / d)
    r_abs = (x_means - k * d).abs()
    if intent in {getattr(RungIntent, "STABILIZE", "stabilize"), "stabilize"}:
        pen = F.relu(r_abs - band) / (band + 1e-12)
    elif intent in {getattr(RungIntent, "HOLD", "hold"), "hold"}:
        tight = 0.5 * band
        pen = F.relu(r_abs - tight) / (tight + 1e-12)
    else:  # SEEK
        pen = F.relu(band - r_abs) / (band + 1e-12)
    return pen.mean()

# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_loop(
    *,
    device=None,
    steps=200,
    vocab=256, d=128, layers=4, heads=4, seq_len=128, fold="grid",
    delta=0.03, capacities=(2, 6, 10, 14), batch=32, use_data=True,
    lr=2e-4, wd=0.01, warmup_frac=0.10, clip_norm=1.0, tv_weight=0.0,
    out: Optional[str] = None, print_every: Optional[int] = None, ui: str = "auto",
    rung_intent: Any = getattr(RungIntent, "STABILIZE", "stabilize"),
    rung_target_k: Optional[int] = None, rung_band: Optional[float] = None,
    rung_loss_weight: float = 0.0,
):
    """
    Main ElementFold training routine (β, γ, ⛔ under rung‑centric feedback).

    Controllers:
      • Supervisor    → smooth coherence controller (β, γ, ⛔ updates).
      • RungController→ higher-level rung phase manager (LOCK / HOLD / SEEK).
    """
    # UI setup
    use_unicode = (_supports_unicode() if ui == "auto" else (ui == "unicode"))
    g = _glyphs(use_unicode)

    # 1) Device
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 2) Build model + “physics” heads
    model = Model(vocab=vocab, d=d, layers=layers, heads=heads,
                  seq_len=seq_len, fold=fold, delta=delta).to(device)
    align = AlignHead(delta).to(device)
    var = VariationalLedger(delta, capacities, tv_weight=float(tv_weight)).to(device)

    # 3) Optimizer and schedule
    opt = build_optimizer(model, lr=lr, wd=wd)
    warmup_steps = max(1, int(warmup_frac * steps))
    scheduler = make_scheduler(opt, warmup_steps=warmup_steps, total_steps=steps, min_lr_scale=0.1)

    # 4) Controllers
    sup = Supervisor()
    band = (float(delta) / 6.0) if (rung_band is None) else float(rung_band)
    rung = RungController(delta=float(delta), intent=rung_intent, k_target=rung_target_k, band=band)

    # 5) Data loader
    if use_data:
        try:
            dl = DataLoaderBuilder(seq_len=seq_len, vocab=vocab, batch=batch).make()
            it = iter(dl)
        except Exception:
            # Fallback: synthetic data
            use_data = False

    # Header
    print(
        f"{g['spin']} ElementFold training  {g['dot']} device={device}  "
        f"{g['dot']} {g['delta']}={_fmt(delta, 5)}  {g['dot']} d={d} L={layers} "
        f"T={seq_len} b={batch}  {g['dot']} steps={steps}  "
        f"{g['dot']} rung={str(rung_intent)} band={_fmt(band, 5)}"
    )

    # 6) Optimization loop
    for step in range(steps):
        # a) Batch
        if use_data:
            try:
                x = next(it).to(device)
            except StopIteration:
                it = iter(dl)  # type: ignore
                x = next(it).to(device)
            except Exception:
                x = torch.randint(0, vocab, (batch, seq_len), device=device)
        else:
            x = torch.randint(0, vocab, (batch, seq_len), device=device)

        # b) Forward
        logits, X = model(x)  # X: ledger/anchored-log values, shape (B,T)
        loss_task = F.cross_entropy(logits.reshape(-1, vocab), x.reshape(-1))

        # Alignment head (per-sample scalar summary)
        xm = X.mean(dim=1)  # (B,)
        caps_t = torch.as_tensor(capacities, device=xm.device)
        loss_align, pos, neg = align(xm, caps_t)

        # Variational ledger energy (respect max available seats)
        maxcap = int(min(X.size(1), int(max(capacities)) if len(capacities) else X.size(1)))
        e = var.energy(X[:, :maxcap])

        # Optional rung penalty
        loss_rung = xm.new_zeros(())
        if rung_loss_weight > 0.0:
            intent_str = str(rung_intent) if isinstance(rung_intent, str) else str(getattr(rung_intent, "name", "stabilize")).lower()
            loss_rung = _rung_penalty(xm, delta=float(delta), band=band, intent=intent_str) * float(rung_loss_weight)

        # Total loss
        loss = loss_task + loss_align + 0.1 * e / (batch * seq_len) + loss_rung

        # c) Backprop
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(clip_grad_norm_(model.parameters(), clip_norm))
        opt.step()
        try:
            scheduler.step()
        except Exception:
            pass

        # d) Telemetry
        tele = measure(xm, float(delta), detail=False)  # returns ASCII + Unicode mirrors
        tele.update({
            "grad_norm": grad_norm,
            "x_mean": float(xm.mean().item()),
            "δ⋆": float(delta),
        })
        # Rung extras (for logs)
        rung_m = _rung_metrics(xm, float(delta))
        k_cur = int(torch.round(xm.mean() / float(delta)).item())
        r_mean = float(rung_m["r"].mean().item())
        tele.update({
            "k_current": k_cur,
            "r": r_mean,
            "r_clicks": float(r_mean / float(delta)),
            "band": float(band),
            "intent": str(rung_intent),
        })

        # e) Controller fusion
        ctrl_sup = sup.update(tele)              # {'beta','gamma','clamp'}
        ctrl_out = rung.update(tele, ctrl_sup)   # possibly refined by rung policy
        if hasattr(model, "apply_control"):
            model.apply_control(**ctrl_out)

        # f) Progress print
        if print_every and ((step + 1) % print_every == 0 or step == steps - 1):
            frac = (step + 1) / max(1, steps)
            lr_now = get_lr(opt)
            bar = _bar(frac, 24, g)
            msg = (
                f"{bar} step {step+1}/{steps}  ℒ={_fmt(float(loss))}  "
                f"{g['kappa']}={_fmt(tele.get('kappa'))}  {g['phalf']}={_fmt(tele.get('p_half'))}  "
                f"{g['grad']}={_fmt(grad_norm, 2)}  {g['bolt']} lr={_fmt(lr_now, 2)}  "
                f"k={k_cur} r(clicks)={_fmt(float(r_mean/float(delta)), 3)}  "
                f"intent={str(rung_intent)}  "
                f"{g['beta']}={_fmt(ctrl_out.get('beta'), 3)} "
                f"{g['gamma']}={_fmt(ctrl_out.get('gamma'), 3)} "
                f"{g['clamp']}={_fmt(ctrl_out.get('clamp'), 3)}"
            )
            if rung_loss_weight > 0.0:
                msg += f"  rung_pen={_fmt(float(loss_rung), 4)}"
            print(msg)

    # 7) Save checkpoint
    if out:
        path = Path(out)
        if path.is_dir() or not path.suffix:
            path.mkdir(parents=True, exist_ok=True)
            path = path / "checkpoint.pt"
        torch.save(model.state_dict(), path)
        print(f"{g['ok']} model saved to {path}")
    else:
        print(f"{g['ok']} training done (no checkpoint path provided)")

    if print_every:
        print(
            "AdamW Cosine/SGDR Clipping TV Denoise  |  "
            "https://doi.org/10.5281/zenodo.17460798  |  "
            "https://doi.org/10.5281/zenodo.17393945  |  "
            "https://doi.org/10.5281/zenodo.17481738"
        )

    return model
