# ElementFold · optim.py
# ============================================================
# Optimizers and schedules — written to explain, not to mystify.
#
# Guiding principles
# ------------------
# • AdamW with clean parameter grouping.
# • Decay only true weights, never biases or norm scales.
# • Optional fused AdamW when runtime supports it.
# • Simple warmup→cosine LR schedule that “just works.”
# • Helpers for clipping, zeroing, counting, and freezing.
# ============================================================

from __future__ import annotations
import math, torch
from torch import nn
from typing import Tuple, Dict, Any, Optional

# ============================================================
# 1. Parameter grouping — decay vs. no-decay
# ============================================================

def _split_decay_groups(model: nn.Module) -> Tuple[list, list]:
    """
    Walk all parameters and split them into two buckets:

      • decay     — true weight matrices that benefit from L2 decay
      • no_decay  — 1-D params (biases, norm scales, etc.)

    Heuristic:
        dim ≤ 1  → no_decay
        '.bias'  → no_decay
        contains {'norm','ln','layernorm','bn','rmsnorm'} → no_decay
    """
    decay, no_decay = [], []
    norm_keys = ("norm","ln","layernorm","bn","rmsnorm")
    for name,p in model.named_parameters():
        if not p.requires_grad: continue
        n=name.lower()
        if p.dim()<=1 or n.endswith(".bias") or any(k in n for k in norm_keys):
            no_decay.append(p)
        else:
            decay.append(p)
    return decay,no_decay

# ============================================================
# 2. AdamW builder (with optional fused kernel)
# ============================================================

def build_optimizer(model:nn.Module, lr:float=2e-4, wd:float=0.01,
                    betas:Tuple[float,float]=(0.9,0.95), eps:float=1e-8,
                    fused:Optional[bool]=None)->torch.optim.Optimizer:
    """
    Build AdamW with two groups (decay / no-decay).
    Automatically enables fused kernels when safe.
    """
    decay,no_decay=_split_decay_groups(model)
    if not (decay or no_decay):
        raise ValueError("No trainable parameters found.")
    groups=[]
    if decay: groups.append({"params":decay,"weight_decay":float(wd)})
    if no_decay: groups.append({"params":no_decay,"weight_decay":0.0})
    extra={"lr":float(lr),"betas":tuple(betas),"eps":float(eps)}
    if fused is not None:
        extra["fused"]=bool(fused)
    else:
        try:
            import inspect
            if torch.cuda.is_available() and \
               "fused" in inspect.signature(torch.optim.AdamW).parameters:
                extra["fused"]=True
        except Exception: pass
    return torch.optim.AdamW(groups,**extra)

# ============================================================
# 3. LR schedule — warmup → cosine decay
# ============================================================

def make_scheduler(optimizer:torch.optim.Optimizer,
                   warmup_steps:int,total_steps:int,
                   min_lr_scale:float=0.1)->torch.optim.lr_scheduler.LambdaLR:
    """
    Linear warmup then smooth cosine decay to `min_lr_scale`.
    """
    warm=max(1,int(warmup_steps))
    total=max(warm+1,int(total_steps))
    min_s=float(min_lr_scale)
    def scale(step:int)->float:
        if step<warm: return float(step+1)/float(warm)
        denom=max(1,total-warm)
        prog=min(1.0,float(step-warm)/float(denom))
        return min_s + 0.5*(1.0-min_s)*(1.0+math.cos(math.pi*prog))
    return torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=scale)

# ============================================================
# 4. Loop helpers — the things you actually use
# ============================================================

def clip_gradients(model:nn.Module,max_norm:float=1.0,norm_type:float=2.0)->float:
    """Global gradient clipping for stability; returns pre-clip norm."""
    params=[p for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not params: return 0.0
    return float(torch.nn.utils.clip_grad_norm_(params,max_norm,norm_type=norm_type))

def zero_grad(optimizer:torch.optim.Optimizer)->None:
    """Zero gradients efficiently (set_to_none=True avoids allocator churn)."""
    optimizer.zero_grad(set_to_none=True)

def get_lr(optimizer:torch.optim.Optimizer)->float:
    """Return current learning rate from first param group."""
    return float(optimizer.param_groups[0]["lr"])

def count_params(model:nn.Module)->Tuple[int,int]:
    """Count (trainable,total) parameters."""
    total=sum(p.numel() for p in model.parameters())
    train=sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(train),int(total)

def set_requires_grad(module:nn.Module,requires_grad:bool)->None:
    """Freeze/unfreeze an entire module tree."""
    flag=bool(requires_grad)
    for p in module.parameters(): p.requires_grad=flag
