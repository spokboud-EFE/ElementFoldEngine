# ElementFold · train.py
# This is the end‑to‑end training loop. Read it like a story:
#   1) We build the model and its two “physics” companions:
#        • AlignHead   — keeps phases aligned on the δ⋆ circle without a temperature.
#        • Variational — keeps seats and blocks equally spaced (convex backbone).
#   2) We create an optimizer (AdamW with clean param groups) and a gentle LR schedule.
#   3) Each step, we fetch a batch (or synthesize tokens), run the model, compute three losses:
#        ℒ_task  (cross‑entropy on tokens),
#        ℒ_align (temperature‑free contrast on the circle),
#        E_ledger (convex spacing energy),
#      then combine them into one total loss (+ optional rung penalty if you enable it).
#   4) We backprop, clip gradients (safety), and step the optimizer and scheduler.
#   5) We measure coherence (κ, p½, margins) and **rung telemetry** (k, residuals).
#   6) We let the **RungController** (center stage) fuse Supervisor hints with rung intent
#      (STABILIZE / SEEK / HOLD) and push β, γ, ⛔ into the model (apply_control).

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from .model import Model
from .align import AlignHead
from .variational import VariationalLedger
from .telemetry import measure
from .optim import build_optimizer, make_scheduler, get_lr
from .control import Supervisor
from .data import DataLoaderBuilder

# ——— Rung control is first‑class: always present ————————————————
from .rung_controller import RungController, RungIntent


# ——————————————————————————————————————————————————————————————————————————
# Small, dependency‑free UI helpers (Unicode‑aware; safe ASCII fallback)
# ——————————————————————————————————————————————————————————————————————————

def _supports_unicode() -> bool:
    enc = getattr(sys.stdout, "encoding", "") or ""
    return "UTF" in enc.upper()


def _glyphs(use_unicode: bool) -> Dict[str, str]:
    if use_unicode:
        return {
            "spin": "⟲", "ok": "✓", "warn": "⚠", "save": "💾",
            "beta": "β", "gamma": "γ", "clamp": "⛔", "delta": "δ⋆",
            "kappa": "κ", "phalf": "p½", "grad": "∥∇∥", "bolt": "⚡",
            "dot": "•", "bar_full": "▰", "bar_empty": "▱",
        }
    else:
        return {
            "spin": "*", "ok": "OK", "warn": "!", "save": "SAVE",
            "beta": "beta", "gamma": "gamma", "clamp": "CLAMP", "delta": "delta*",
            "kappa": "kappa", "phalf": "p_half", "grad": "||grad||", "bolt": ">",
            "dot": "-", "bar_full": "#", "bar_empty": "-",
        }


def _bar(frac: float, width: int, g: Dict[str, str]) -> str:
    frac = max(0.0, min(1.0, float(frac)))
    full = int(round(frac * width))
    empty = width - full
    return "[" + (g["bar_full"] * full) + (g["bar_empty"] * empty) + f"] {int(frac * 100):3d}%"


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


# ——————————————————————————————————————————————————————————————————————————
# Rung helpers — compute k and residuals (non‑expert narration in comments)
# ——————————————————————————————————————————————————————————————————————————
def _rung_metrics(x: torch.Tensor, delta: float) -> Dict[str, torch.Tensor]:
    """
    Given a tensor of anchored‑log values x (any shape), compute:
      • k = round(x/δ⋆)               (nearest integer rung)
      • r = x - k·δ⋆                  (residual toward the current rung)
      • r_clicks = r / δ⋆             (residual in 'clicks' for readability)
    We do this element‑wise; callers can mean() if they want a single batch number.
    """
    d = float(delta)
    k = torch.round(x / d)
    r = x - k * d
    r_clicks = r / d
    return {"k": k, "r": r, "r_clicks": r_clicks}


def _rung_penalty(x_means: torch.Tensor, delta: float, band: float, intent: str) -> torch.Tensor:
    """
    Optional training penalty around rungs:
      • STABILIZE: penalize being outside the acceptance band (|r| > band)
      • HOLD     : like STABILIZE but with a tighter band (0.5×)
      • SEEK     : penalize being too close to rungs (|r| < band) to encourage safe misalignment
    Returns a scalar penalty (mean over batch).
    """
    d = float(delta)
    # residual magnitude per example
    k = torch.round(x_means / d)
    r_abs = (x_means - k * d).abs()

    if intent == RungIntent.STABILIZE or intent == "stabilize":
        penalty = F.relu(r_abs - band) / (band + 1e-12)  # outside → positive; inside → 0
    elif intent == RungIntent.HOLD or intent == "hold":
        tight = 0.5 * band
        penalty = F.relu(r_abs - tight) / (tight + 1e-12)
    else:  # SEEK / disalign: inside the band is penalized so model sits safely away
        penalty = F.relu(band - r_abs) / (band + 1e-12)

    return penalty.mean()


# ——————————————————————————————————————————————————————————————————————————
# Training loop (friendly logging + rung‑centric control + optional rung penalty)
# ——————————————————————————————————————————————————————————————————————————
def train_loop(
    # Core geometry / model
    device=None,
    steps=200,
    vocab=256, d=128, layers=4, heads=4,
    seq_len=128, fold='grid',
    delta=0.03,
    capacities=(2, 6, 10, 14),
    batch=32, use_data=True,

    # Optimizer & regularization
    lr=2e-4, wd=0.01,
    warmup_frac=0.1,
    clip_norm=1.0,
    tv_weight=0.0,

    # Output & logging
    out: str | None = None,           # file or dir; if dir, writes checkpoint.pt
    print_every: int | None = None,   # e.g. 50 → progress line every 50 steps; None = silent
    ui: str = "auto",                 # "unicode" | "ascii" | "auto"

    # ★ Rung control (center stage)
    rung_intent: str | RungIntent = RungIntent.STABILIZE,  # "stabilize" | "seek" | "hold"
    rung_target_k: int | None = None, # optional target rung index for HOLD/SEEK strategies
    rung_band: float | None = None,   # acceptance half‑band in X; default δ⋆/6 if None
    rung_loss_weight: float = 0.0,    # 0 = off; >0 adds a small rung penalty to the total loss
):
    r"""
    ⟲ ElementFold training loop — now **rung‑centric**.

    Why the rung controller here?
    -----------------------------
    ElementFold’s Supervisor keeps the model coherent (β exposure, γ damping, ⛔ clamp).
    The **RungController** sits on top to bias where we sit on the click ladder:
      • STABILIZE — stay near the current rung (good for accuracy/consistency).
      • HOLD      — like stabilize but with tighter band (for metrology‑grade runs).
      • SEEK      — stay *outside* the band (useful when you want safe disalignment).

    What is the “band”?
    -------------------
    The acceptance half‑band is the tolerance around a rung center, in anchored‑log units X.
    A classic, conservative choice is **band = δ⋆ / 6**, leaving a 2× bigger guard to the mid‑step.

    Optional rung penalty
    ---------------------
    If you want the loss itself to “feel” the rung behavior, set `rung_loss_weight > 0`.
    • STABILIZE/HOLD: penalize |r| > band (be inside).
    • SEEK: penalize |r| < band (be outside).
    This is gentle shaping; most of the steering still happens via β/γ/⛔.

    Unicode vs ASCII output
    -----------------------
    If your terminal can’t render Unicode, the logger automatically falls back to ASCII.

    References (for the curious)
    ----------------------------
    • AdamW (Decoupled Weight Decay): https://arxiv.org/abs/1711.05101
    • Cosine annealing / SGDR:       https://arxiv.org/abs/1608.03983
    • Gradient clipping (theory):    https://proceedings.mlr.press/v28/pascanu13.html
    • Total variation denoising:     https://en.wikipedia.org/wiki/Total_variation_denoising
    • von Mises distribution (κ):    https://en.wikipedia.org/wiki/Von_Mises_distribution
    """

    # ——— UI setup ——————————————————————————————————————————————
    use_unicode = (_supports_unicode() if ui == "auto" else (ui == "unicode"))
    g = _glyphs(use_unicode)
    t0 = time.time()

    # ——— 1) Device ————————————————————————————————————————————
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ——— 2) Build model and companions ——————————————————————
    model = Model(vocab=vocab, d=d, layers=layers, heads=heads, seq_len=seq_len, fold=fold, delta=delta).to(device)
    align = AlignHead(delta).to(device)
    var = VariationalLedger(delta, capacities, tv_weight=float(tv_weight)).to(device)

    # ——— 3) Optimizer + warmup/cosine ——————————————————————
    opt = build_optimizer(model, lr=lr, wd=wd)
    warmup_steps = max(1, int(warmup_frac * steps))
    scheduler = make_scheduler(opt, warmup_steps=warmup_steps, total_steps=steps, min_lr_scale=0.1)

    # ——— 4) Supervisor + RungController ————————————————————
    sup = Supervisor()  # baseline coherence controller
    band = float(delta) / 6.0 if rung_band is None else float(rung_band)
    rung = RungController(
        delta=float(delta),
        intent=rung_intent,
        k_target=rung_target_k,
        band=band,
    )

    # ——— 5) Data stream (or synthetic) ————————————————————
    if use_data:
        dl = DataLoaderBuilder(seq_len=seq_len, vocab=vocab, batch=batch).make()
        it = iter(dl)

    # ——— Header ——————————————————————————————————————————————
    print(
        f"{g['spin']} ElementFold training  "
        f"{g['dot']} device={device}  {g['dot']} {g['delta']}={_fmt(delta, 5)}  "
        f"{g['dot']} d={d} L={layers} T={seq_len} b={batch}  "
        f"{g['dot']} steps={steps}  {g['dot']} rung={str(rung_intent)} band={_fmt(band, 5)}"
    )

    # ——— 6) Optimization loop ——————————————————————————————
    for step in range(steps):
        # 6.a) Batches (rewind on exhaustion)
        if use_data:
            try:
                x = next(it).to(device)  # (B,T) int64
            except StopIteration:
                it = iter(dl)
                x = next(it).to(device)
        else:
            x = torch.randint(0, vocab, (batch, seq_len), device=device)

        # 6.b) Forward — logits and ledger X (anchored log per token)
        logits, X = model(x)  # logits: (B,T,V), X: (B,T)

        # 6.c) Core losses
        loss_task = F.cross_entropy(logits.reshape(-1, vocab), x.reshape(-1))

        caps_t = torch.as_tensor(capacities, device=device)
        loss_align, pos, neg = align(X.mean(dim=1), caps_t)

        # Variational energy over first max(capacities) seats (clamped by T)
        maxcap = int(min(X.size(1), int(max(capacities)) if len(capacities) else X.size(1)))
        e = var.energy(X[:, :maxcap])

        # (optional) Rung penalty: operate on batch‑means of X
        loss_rung = torch.tensor(0.0, device=device)
        if rung_loss_weight > 0.0:
            x_means = X.mean(dim=1)  # (B,)
            intent_str = str(rung_intent) if not isinstance(rung_intent, str) else rung_intent
            loss_rung = _rung_penalty(x_means, delta=float(delta), band=band, intent=intent_str) * float(rung_loss_weight)

        # Combine: task primary; align speaks; variational small; rung optional
        loss = loss_task + 1.0 * loss_align + 0.1 * e / (batch * seq_len) + loss_rung

        # 6.d) Backprop — safety rails
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(clip_grad_norm_(model.parameters(), clip_norm))
        opt.step()
        scheduler.step()

        # 6.e) Telemetry — coherence + rung
        x_batch_mean = X.mean(dim=1)                 # (B,)
        tele = measure(x_batch_mean, float(delta), detail=False)
        tele["grad_norm"] = grad_norm
        tele["x_mean"] = float(x_batch_mean.mean().item())

        # Rung metrics (report mean over batch for readable numbers)
        rung_m = _rung_metrics(x_batch_mean, float(delta))
        k_cur = int(torch.round(x_batch_mean.mean() / float(delta)).item())
        r_mean = float(rung_m["r"].mean().item())
        r_clicks_mean = float(rung_m["r_clicks"].mean().item())

        tele["k_current"] = k_cur
        tele["r"] = r_mean
        tele["r_clicks"] = r_clicks_mean
        tele["band"] = float(band)
        tele["intent"] = str(rung_intent)

        # 6.f) Supervisory fusion — Supervisor → RungController → model
        ctrl_sup = sup.update(tele)       # {'beta','gamma','clamp'} suggestions for stability
        ctrl_out = rung.update(tele, ctrl_sup)  # rung‑centric override/fuse
        if hasattr(model, "apply_control"):
            model.apply_control(beta=ctrl_out["beta"], gamma=ctrl_out["gamma"], clamp=ctrl_out["clamp"])

        # 6.g) Friendly progress
        if print_every is not None and ((step + 1) % print_every == 0 or step == steps - 1):
            frac = (step + 1) / max(1, steps)
            lr_now = get_lr(opt)
            bar = _bar(frac, width=24, g=g)
            msg = (
                f"{bar}  step {step+1}/{steps}  "
                f"ℒ={_fmt(float(loss))}  "
                f"{g['kappa']}={_fmt(tele.get('kappa'))}  "
                f"{g['phalf']}={_fmt(tele.get('p_half'))}  "
                f"{g['grad']}={_fmt(grad_norm, 2)}  "
                f"{g['bolt']} lr={_fmt(lr_now, 2)}  "
                f"k={k_cur}  r(clicks)={_fmt(r_clicks_mean, 3)}  "
                f"intent={str(rung_intent)}  "
                f"{g['beta']}={_fmt(ctrl_out.get('beta'), 3)}  {g['gamma']}={_fmt(ctrl_out.get('gamma'), 3)}  {g['clamp']}={_fmt(ctrl_out.get('clamp'), 3)}"
            )
            # Show rung penalty only when active
            if rung_loss_weight > 0.0:
                msg += f"  rung_pen={_fmt(float(loss_rung), 4)}"
            print(msg)

    # ——— Save checkpoint if requested —————————————————————————
    if out is not None:
        path = Path(out)
        if path.is_dir() or not path.suffix:
            path.mkdir(parents=True, exist_ok=True)
            path = path / "checkpoint.pt"
        torch.save(model.state_dict(), path)
        print(f"{g['ok']} model saved to {path}")
    else:
        print(f"{g['ok']} training done (no checkpoint path provided)")

    # ——— Epilogue ——————————————————————————————————————————————
    if print_every is not None:
        print(
            "Refs: AdamW https://arxiv.org/abs/1711.05101  |  Cosine/SGDR https://arxiv.org/abs/1608.03983  |  "
            "Clipping https://proceedings.mlr.press/v28/pascanu13.html  |  TV https://en.wikipedia.org/wiki/Total_variation_denoising"
        )

    return model
