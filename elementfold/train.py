# ElementFold · train.py
# ============================================================
# ElementFold training loop — “rung-centric” orchestration.
#
# The model learns while staying coherent on the δ★ circle:
#   • Supervisor keeps β (exposure), γ (damping), and ⛔ (clamp) in safe ranges.
#   • RungController handles discrete rung behavior (LOCK/HOLD/SEEK).
#   • Telemetry reports κ (coherence) and p½ (barrier proximity).
#
# This file glues everything together: model, optimizer, telemetry,
# control feedback, and friendly progress printing.
# ============================================================

from __future__ import annotations
import sys, time
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from .core.model import Model
from .align import AlignHead
from .core.variational import VariationalLedger
from .core.telemetry import measure
from .optim import build_optimizer, make_scheduler, get_lr
from .core.control import Supervisor
from .data import DataLoaderBuilder
from .rung_controller import RungController, RungIntent

# ============================================================
# 1. UI helpers (Unicode-aware with ASCII fallback)
# ============================================================

def _supports_unicode() -> bool:
    enc = getattr(sys.stdout, "encoding", "") or ""
    return "UTF" in enc.upper()

def _glyphs(use_unicode: bool) -> Dict[str,str]:
    return ({
        "spin":"⟲","ok":"✓","warn":"⚠","save":"💾","beta":"β","gamma":"γ","clamp":"⛔",
        "delta":"δ⋆","kappa":"κ","phalf":"p½","grad":"∥∇∥","bolt":"⚡","dot":"•",
        "bar_full":"▰","bar_empty":"▱"
    } if use_unicode else {
        "spin":"*","ok":"OK","warn":"!","save":"SAVE","beta":"beta","gamma":"gamma",
        "clamp":"CLAMP","delta":"delta*","kappa":"kappa","phalf":"p_half","grad":"||grad||",
        "bolt":">","dot":"-","bar_full":"#","bar_empty":"-"
    })

def _bar(frac: float, width: int, g: Dict[str,str]) -> str:
    frac = max(0.0,min(1.0,float(frac)))
    full = int(round(frac*width))
    return "[" + g["bar_full"]*full + g["bar_empty"]*(width-full) + f"] {int(frac*100):3d}%"

def _fmt(x: float|int|None, digits:int=4) -> str:
    if x is None: return "—"
    if isinstance(x,int): return f"{x}"
    try:
        if abs(x)>=1e3 or (0<abs(x)<1e-3): return f"{x:.2e}"
        return f"{x:.{digits}f}"
    except Exception: return str(x)

# ============================================================
# 2. Helpers for rung metrics
# ============================================================

def _rung_metrics(x: torch.Tensor, delta: float) -> Dict[str, torch.Tensor]:
    """Compute rung index and residual for a batch of anchored-log values."""
    d = float(delta)
    k = torch.round(x / d)
    r = x - k * d
    return {"k":k,"r":r,"r_clicks":r/d}

def _rung_penalty(x_means: torch.Tensor, delta: float, band: float, intent: str) -> torch.Tensor:
    """Optional gentle penalty encouraging/avoiding proximity to rungs."""
    d = float(delta)
    k = torch.round(x_means / d)
    r_abs = (x_means - k*d).abs()
    if intent in {RungIntent.STABILIZE,"stabilize"}:
        pen = F.relu(r_abs - band)/(band+1e-12)
    elif intent in {RungIntent.HOLD,"hold"}:
        tight = 0.5*band
        pen = F.relu(r_abs - tight)/(tight+1e-12)
    else:  # SEEK
        pen = F.relu(band - r_abs)/(band+1e-12)
    return pen.mean()

# ============================================================
# 3. Training loop
# ============================================================

def train_loop(
    device=None, steps=200,
    vocab=256, d=128, layers=4, heads=4, seq_len=128, fold='grid',
    delta=0.03, capacities=(2,6,10,14), batch=32, use_data=True,
    lr=2e-4, wd=0.01, warmup_frac=0.1, clip_norm=1.0, tv_weight=0.0,
    out: str|None=None, print_every:int|None=None, ui:str="auto",
    rung_intent:RungIntent|str=RungIntent.STABILIZE,
    rung_target_k:int|None=None, rung_band:float|None=None,
    rung_loss_weight:float=0.0,
):
    """
    Main ElementFold training routine (β,γ,⛔ under rung-centric feedback).

    Controllers:
      • Supervisor  → smooth coherence controller (β,γ,⛔ updates).
      • RungController → higher-level rung phase manager (LOCK/HOLD/SEEK).
    """
    # --------------------------------------------------------
    # UI setup
    # --------------------------------------------------------
    use_unicode = (_supports_unicode() if ui=="auto" else ui=="unicode")
    g = _glyphs(use_unicode)
    t0 = time.time()

    # --------------------------------------------------------
    # 1) Device
    # --------------------------------------------------------
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------
    # 2) Build model + “physics” heads
    # --------------------------------------------------------
    model = Model(vocab=vocab,d=d,layers=layers,heads=heads,
                  seq_len=seq_len,fold=fold,delta=delta).to(device)
    align = AlignHead(delta).to(device)
    var = VariationalLedger(delta,capacities,tv_weight=float(tv_weight)).to(device)

    # --------------------------------------------------------
    # 3) Optimizer and schedule
    # --------------------------------------------------------
    opt = build_optimizer(model,lr=lr,wd=wd)
    warmup_steps = max(1,int(warmup_frac*steps))
    scheduler = make_scheduler(opt,warmup_steps=warmup_steps,total_steps=steps,min_lr_scale=0.1)

    # --------------------------------------------------------
    # 4) Controllers
    # --------------------------------------------------------
    sup = Supervisor()
    band = float(delta)/6 if rung_band is None else float(rung_band)
    rung = RungController(delta=float(delta),intent=rung_intent,
                          k_target=rung_target_k,band=band)

    # --------------------------------------------------------
    # 5) Data loader
    # --------------------------------------------------------
    if use_data:
        dl = DataLoaderBuilder(seq_len=seq_len,vocab=vocab,batch=batch).make()
        it = iter(dl)

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------
    print(f"{g['spin']} ElementFold training  {g['dot']} device={device}  "
          f"{g['dot']} {g['delta']}={_fmt(delta,5)}  {g['dot']} d={d} L={layers} "
          f"T={seq_len} b={batch}  {g['dot']} steps={steps}  "
          f"{g['dot']} rung={str(rung_intent)} band={_fmt(band,5)}")

    # --------------------------------------------------------
    # 6) Optimization loop
    # --------------------------------------------------------
    for step in range(steps):
        # a) Batch
        if use_data:
            try:
                x = next(it).to(device)
            except StopIteration:
                it = iter(dl); x = next(it).to(device)
        else:
            x = torch.randint(0,vocab,(batch,seq_len),device=device)

        # b) Forward
        logits,X = model(x)          # X: anchored log values
        loss_task = F.cross_entropy(logits.reshape(-1,vocab),x.reshape(-1))
        caps_t = torch.as_tensor(capacities,device=device)
        loss_align,pos,neg = align(X.mean(dim=1),caps_t)
        maxcap = int(min(X.size(1),int(max(capacities)) if len(capacities) else X.size(1)))
        e = var.energy(X[:,:maxcap])

        # optional rung penalty
        loss_rung = torch.tensor(0.0,device=device)
        if rung_loss_weight>0:
            xm = X.mean(dim=1)
            intent_str = str(rung_intent) if not isinstance(rung_intent,str) else rung_intent
            loss_rung = _rung_penalty(xm,delta=float(delta),band=band,intent=intent_str)*rung_loss_weight

        # combine
        loss = loss_task + loss_align + 0.1*e/(batch*seq_len) + loss_rung

        # c) Backprop
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(clip_grad_norm_(model.parameters(),clip_norm))
        opt.step(); scheduler.step()

        # d) Telemetry
        xm = X.mean(dim=1)
        tele = measure(xm,float(delta),detail=False)
        tele.update({"grad_norm":grad_norm,"x_mean":float(xm.mean().item()),"δ⋆":float(delta)})
        if "kappa" in tele and "κ" not in tele: tele["κ"]=tele["kappa"]
        if "p_half" in tele and "p½" not in tele: tele["p½"]=tele["p_half"]
        rung_m = _rung_metrics(xm,float(delta))
        k_cur = int(torch.round(xm.mean()/float(delta)).item())
        r_mean = float(rung_m["r"].mean().item())
        tele.update({"k_current":k_cur,"r":r_mean,"r_clicks":float(r_mean/float(delta)),
                     "band":float(band),"intent":str(rung_intent)})

        # e) Controller fusion
        ctrl_sup = sup.update(tele)
        ctrl_out = rung.update(tele,ctrl_sup)
        if hasattr(model,"apply_control"):
            model.apply_control(**ctrl_out)

        # f) Progress print
        if print_every and ((step+1)%print_every==0 or step==steps-1):
            frac=(step+1)/max(1,steps)
            lr_now=get_lr(opt)
            bar=_bar(frac,24,g)
            msg=(f"{bar} step {step+1}/{steps}  ℒ={_fmt(float(loss))}  "
                 f"{g['kappa']}={_fmt(tele.get('kappa'))}  {g['phalf']}={_fmt(tele.get('p_half'))}  "
                 f"{g['grad']}={_fmt(grad_norm,2)}  {g['bolt']} lr={_fmt(lr_now,2)}  "
                 f"k={k_cur} r(clicks)={_fmt(float(r_mean/float(delta)),3)}  "
                 f"intent={str(rung_intent)}  "
                 f"{g['beta']}={_fmt(ctrl_out.get('beta'),3)} "
                 f"{g['gamma']}={_fmt(ctrl_out.get('gamma'),3)} "
                 f"{g['clamp']}={_fmt(ctrl_out.get('clamp'),3)}")
            if rung_loss_weight>0: msg+=f"  rung_pen={_fmt(float(loss_rung),4)}"
            print(msg)

    # --------------------------------------------------------
    # 7) Save checkpoint
    # --------------------------------------------------------
    if out:
        path=Path(out)
        if path.is_dir() or not path.suffix:
            path.mkdir(parents=True,exist_ok=True)
            path=path/"checkpoint.pt"
        torch.save(model.state_dict(),path)
        print(f"{g['ok']} model saved to {path}")
    else:
        print(f"{g['ok']} training done (no checkpoint path provided)")

    if print_every:
        print("Refs: AdamW https://arxiv.org/abs/1711.05101  |  "
              "Cosine/SGDR https://arxiv.org/abs/1608.03983  |  "
              "Clipping https://proceedings.mlr.press/v28/pascanu13.html  |  "
              "TV https://en.wikipedia.org/wiki/Total_variation_denoising")

    return model
