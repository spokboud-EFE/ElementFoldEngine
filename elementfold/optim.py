# ElementFold · optim.py
# ============================================================
# Optimizers and schedules — written to explain, not to mystify.
#
# What this module provides
# -------------------------
# • build_optimizer(...)  → AdamW with clean param grouping
# • make_scheduler(...)   → warmup → cosine LR schedule
# • clip_gradients(...)   → stable global grad clipping
# • zero_grad(...)        → efficient gradient zeroing
# • get_lr(...)           → read current LR from optimizer
# • count_params(...)     → (trainable, total) parameter counts
# • set_requires_grad(...)→ freeze / unfreeze a module tree
#
# Design choices
# --------------
# • Decay only *true* weights (matrices). Never decay biases,
#   normalization scales, or embeddings (common practice).
# • Try fused AdamW and foreach stepping when the runtime supports it.
# • No project-internal imports; pure PyTorch + stdlib.
# ============================================================

from __future__ import annotations

import math
import inspect
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn

__all__ = [
    "build_optimizer",
    "make_scheduler",
    "clip_gradients",
    "zero_grad",
    "get_lr",
    "count_params",
    "set_requires_grad",
]

# ---------------------------------------------------------------------
# 1) Parameter grouping — decay vs. no-decay
# ---------------------------------------------------------------------


def _is_norm_like(name: str) -> bool:
    """
    Identify normalization parameters by name (case-insensitive).
    We avoid weight decay on these (LayerNorm / RMSNorm / BatchNorm, etc.).
    """
    n = name.lower()
    return any(tag in n for tag in ("norm", "layernorm", "rmsnorm", "bn", "batchnorm", "groupnorm", "instancenorm"))


def _is_embedding_like(name: str) -> bool:
    """
    Identify embeddings (token/positional) by name.
    Common practice is to avoid L2 weight decay on embedding tables.
    """
    n = name.lower()
    return "embedding" in n or "embeddings" in n or "pos_embedding" in n or "positional_embedding" in n


def _split_decay_groups(model: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """
    Walk all parameters and split them into two buckets:

      • decay     — true weight matrices that benefit from L2 decay
      • no_decay  — 1-D params (biases, norm scales), embeddings, etc.

    Heuristics:
      • dim ≤ 1                     → no_decay
      • name endswith '.bias'       → no_decay
      • norm-like (LayerNorm, BN)   → no_decay
      • embedding-like              → no_decay
      • otherwise                   → decay
    """
    decay: List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() <= 1 or name.endswith(".bias") or _is_norm_like(name) or _is_embedding_like(name):
            no_decay.append(p)
        else:
            decay.append(p)
    return decay, no_decay


# ---------------------------------------------------------------------
# 2) AdamW builder (fused / foreach when available)
# ---------------------------------------------------------------------


def _supports_flag(optim_cls, flag: str) -> bool:
    """Return True if `optim_cls` constructor accepts a parameter named `flag`."""
    try:
        sig = inspect.signature(optim_cls)
        return flag in sig.parameters
    except Exception:
        return False


def build_optimizer(
    model: nn.Module,
    lr: float = 2e-4,
    wd: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    fused: Optional[bool] = None,
    foreach: Optional[bool] = None,
) -> torch.optim.Optimizer:
    """
    Build AdamW with two param groups (decay / no-decay) and best-available kernels.

    Args:
        model:  nn.Module with named_parameters()
        lr:     learning rate
        wd:     weight decay for 'decay' group (no-decay group uses 0.0)
        betas:  Adam betas
        eps:    Adam epsilon
        fused:  if True/False, force-enable/disable fused AdamW; if None, auto-detect
        foreach:if True/False, force-enable/disable foreach stepping; if None, auto-detect

    Returns:
        torch.optim.Optimizer (AdamW)
    """
    decay, no_decay = _split_decay_groups(model)
    if not (decay or no_decay):
        raise ValueError("No trainable parameters found.")

    param_groups = []
    if decay:
        param_groups.append({"params": decay, "weight_decay": float(wd)})
    if no_decay:
        param_groups.append({"params": no_decay, "weight_decay": 0.0})

    # Prepare extra kwargs guarded by signature checks, so we work across PyTorch versions.
    extra = {"lr": float(lr), "betas": tuple(betas), "eps": float(eps)}

    # foreach support (PyTorch 2.0+)
    if foreach is not None and _supports_flag(torch.optim.AdamW, "foreach"):
        extra["foreach"] = bool(foreach)
    elif foreach is None and _supports_flag(torch.optim.AdamW, "foreach"):
        # Safe default: enable foreach when available; modest speedup on many setups.
        extra["foreach"] = True

    # fused support (CUDA + recent PyTorch)
    if fused is not None and _supports_flag(torch.optim.AdamW, "fused"):
        extra["fused"] = bool(fused)
    elif fused is None and _supports_flag(torch.optim.AdamW, "fused"):
        # Enable fused if CUDA is present; falls back internally if not applicable.
        extra["fused"] = torch.cuda.is_available()

    return torch.optim.AdamW(param_groups, **extra)


# ---------------------------------------------------------------------
# 3) LR schedule — warmup → cosine decay
# ---------------------------------------------------------------------


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_scale: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Linear warmup to LR, then smooth cosine decay down to `min_lr_scale * base_lr`.

    Args:
        optimizer:    the optimizer to schedule
        warmup_steps: number of warmup steps (clamped to ≥ 1)
        total_steps:  total number of steps (clamped to > warmup)
        min_lr_scale: final LR as a fraction of base LR (e.g., 0.1 → 10% of base)

    Returns:
        LambdaLR scheduler
    """
    warm = max(1, int(warmup_steps))
    total = max(warm + 1, int(total_steps))
    min_s = float(min_lr_scale)

    def scale(step: int) -> float:
        # Warmup
        if step < warm:
            return float(step + 1) / float(warm)
        # Cosine decay to min_s
        denom = max(1, total - warm)
        prog = min(1.0, float(step - warm) / float(denom))
        return min_s + 0.5 * (1.0 - min_s) * (1.0 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scale)


# ---------------------------------------------------------------------
# 4) Loop helpers — the things you actually use
# ---------------------------------------------------------------------


def clip_gradients(model: nn.Module, max_norm: float = 1.0, norm_type: float = 2.0) -> float:
    """
    Global gradient clipping for stability; returns the pre-clip total norm.

    Notes:
      • Skips parameters with no grad.
      • Uses torch.nn.utils.clip_grad_norm_ under the hood.
    """
    params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not params:
        return 0.0
    return float(torch.nn.utils.clip_grad_norm_(params, max_norm, norm_type=norm_type))


def zero_grad(optimizer: torch.optim.Optimizer) -> None:
    """Zero gradients efficiently (set_to_none=True avoids allocator churn)."""
    optimizer.zero_grad(set_to_none=True)


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Return current learning rate from the first param group."""
    return float(optimizer.param_groups[0]["lr"])


def count_params(model: nn.Module) -> Tuple[int, int]:
    """Count (trainable, total) parameters."""
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(train), int(total)


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    """Freeze/unfreeze an entire module tree."""
    flag = bool(requires_grad)
    for p in module.parameters():
        p.requires_grad = flag
