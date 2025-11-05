# ElementFold · optim.py
# Optimizers, gradient safety, and a gentle LR schedule — written so you can
# read it top‑to‑bottom and know *why* each piece exists.
#
# Design principles:
#   • AdamW with sane defaults (β₁, β₂) and accurate parameter grouping.
#   • Decay only true weights (matrices/filters); never decay biases or norm scales.
#   • Optional fused AdamW when available (free speed on recent CUDA builds).
#   • A tiny warmup→cosine schedule that “just works” with per‑step .step().
#   • Utilities for clipping, zero‑grad, param counts, and quick freezing.

from __future__ import annotations

import math
from typing import Tuple, Dict, Any, Optional

import torch
from torch import nn


# ──────────────────────────────────────────────────────────────────────────────
# Parameter grouping (decay vs. no‑decay)
# ──────────────────────────────────────────────────────────────────────────────

def _split_decay_groups(model: nn.Module) -> Tuple[list, list]:
    """
    Walk all named parameters and split them into two buckets:
      • decay     — true weights (matrices/filters) that benefit from weight decay
      • no_decay  — 1‑D params (biases, LayerNorm/RMSNorm scales, etc.) that should *not* decay

    Heuristic (simple, robust):
      - 1D params (p.dim() <= 1) → no_decay
      - Names ending with '.bias' → no_decay
      - Names containing norm keywords ('norm','ln','layernorm','bn','rmsnorm') → no_decay
      - Everything else → decay
    """
    decay, no_decay = [], []
    norm_keywords = ("norm", "ln", "layernorm", "bn", "rmsnorm")

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue  # skip frozen params entirely
        n = name.lower()
        if p.dim() <= 1 or n.endswith(".bias") or any(k in n for k in norm_keywords):
            no_decay.append(p)
        else:
            decay.append(p)
    return decay, no_decay


# ──────────────────────────────────────────────────────────────────────────────
# Optimizer builder (AdamW with optional fused kernels)
# ──────────────────────────────────────────────────────────────────────────────

def build_optimizer(
    model: nn.Module,
    lr: float = 2e-4,
    wd: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    fused: Optional[bool] = None,
) -> torch.optim.Optimizer:
    """
    Create an AdamW optimizer with two parameter groups (decay / no‑decay).

    Args:
      model:  the module providing parameters
      lr:     base learning rate
      wd:     weight decay for the 'decay' group (the 'no_decay' group uses 0)
      betas:  AdamW momentum coefficients (β₁, β₂)
      eps:    numerical epsilon
      fused:  if True/False, force fused AdamW on/off; if None, auto‑enable when supported

    Returns:
      torch.optim.Optimizer ready for training.
    """
    decay, no_decay = _split_decay_groups(model)

    groups: list[Dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": float(wd)})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    if not groups:
        raise ValueError("build_optimizer: model exposes no trainable parameters")

    # Prepare kwargs; softly opt‑in to fused AdamW if the runtime supports it.
    extra: Dict[str, Any] = {"lr": float(lr), "betas": tuple(betas), "eps": float(eps)}
    if fused is not None:
        extra["fused"] = bool(fused)
    else:
        # Auto‑detect fused support only when CUDA is available and the signature allows it.
        try:
            import inspect
            if torch.cuda.is_available() and "fused" in inspect.signature(torch.optim.AdamW).parameters:
                extra["fused"] = True
        except Exception:
            pass  # older PyTorch or CPU‑only builds — just ignore

    opt = torch.optim.AdamW(groups, **extra)
    return opt


# ──────────────────────────────────────────────────────────────────────────────
# Learning‑rate schedule (warmup → cosine)
# ──────────────────────────────────────────────────────────────────────────────

def make_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_scale: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    A gentle schedule for stable training:

        step < warmup:   lr ← lr_base * linear(0 → 1)
        else:            lr ← lr_base * cosine(1 → min_lr_scale)

    We return a LambdaLR; call `scheduler.step()` once per iteration.

    Args:
      optimizer:      an AdamW instance (or any optimizer)
      warmup_steps:   number of steps to ramp 0 → 1
      total_steps:    total planned steps (used to size the cosine tail)
      min_lr_scale:   final fraction of lr at the end (e.g., 0.1 → 10% of base lr)
    """
    warm = max(1, int(warmup_steps))                 # ensure at least one warmup step
    total = max(warm + 1, int(total_steps))          # ensure a nonempty cosine phase
    min_s = float(min_lr_scale)

    def scale(step: int) -> float:
        # Note: LambdaLR calls this with a 0‑based step counter.
        if step < warm:
            # linear ramp from 0 → 1 across 'warm' calls
            return float(step + 1) / float(warm)
        # cosine decay from 1 → min_s over the remaining steps
        denom = max(1, total - warm)
        prog = min(1.0, float(step - warm) / float(denom))
        # cos goes 1→−1 as prog goes 0→1; we remap to 1→min_s
        return min_s + 0.5 * (1.0 - min_s) * (1.0 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scale)


# ──────────────────────────────────────────────────────────────────────────────
# Small utilities you actually use in loops
# ──────────────────────────────────────────────────────────────────────────────

def clip_gradients(model: nn.Module, max_norm: float = 1.0, norm_type: float = 2.0) -> float:
    """
    Clamp global gradient norm to keep updates stable.
    Returns the total (pre‑clip) norm — handy for telemetry.
    """
    params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not params:
        return 0.0
    return float(torch.nn.utils.clip_grad_norm_(params, max_norm, norm_type=norm_type))


def zero_grad(optimizer: torch.optim.Optimizer) -> None:
    """
    Reset gradients without filling memory with zeros (set_to_none=True).
    This is both faster and friendlier to the allocator.
    """
    optimizer.zero_grad(set_to_none=True)


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Read the *first* param group’s learning rate (all groups share lr in our setup).
    """
    return float(optimizer.param_groups[0]["lr"])


def count_params(model: nn.Module) -> Tuple[int, int]:
    """
    Count parameters for quick size telemetry.
    Returns (trainable, total) as integers.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(trainable), int(total)


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    """
    Freeze or unfreeze an entire module tree.
    Useful for staged training or adapter fine‑tuning.
    """
    flag = bool(requires_grad)
    for p in module.parameters():
        p.requires_grad = flag
