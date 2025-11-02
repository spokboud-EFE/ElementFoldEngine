# ElementFold · optim.py
# This file sets up optimizers, gradient clipping, and a simple learning‑rate scheduler.
# The comments are written so a non‑expert can follow *why* each line exists.
# Design goals:
#   • Stable by default: AdamW with sensible betas; no weight decay on biases/norms.
#   • Parameter grouping: decay only true weight matrices/filters; keep scale params clean.
#   • Optional fused AdamW if the PyTorch build supports it (a free speed boost on CUDA).
#   • Utilities you actually use in loops: gradient clipping, warmup+cosine schedule, zero‑grad.

import math                      # For π-free cosine; we just need cos()
import torch                     # Tensors and optimizers
from typing import Iterable, Tuple, Dict, Any, Callable  # Light type hints for readability


def _split_decay_groups(model: torch.nn.Module) -> Tuple[list, list]:
    """
    Walk over named parameters and split them into two buckets:
      • decay:     true weights (matrices/filters) that benefit from weight decay
      • no_decay:  biases and scale parameters (e.g., LayerNorm γ) that should not decay

    Heuristic:
      - 1D params (bias, norm scale) → no_decay
      - Names ending with '.bias'     → no_decay
      - Obvious norm layers (layernorm, ln, norm, bn, rmsnorm) → no_decay
    """
    decay, no_decay = [], []                     # Two empty buckets
    norm_keywords = ("norm", "ln", "layernorm", "bn", "rmsnorm")  # Common scale/bias holders

    for name, p in model.named_parameters():     # Iterate all (name, tensor) pairs
        if not p.requires_grad:                  # Frozen params are skipped entirely
            continue
        n = name.lower()                         # Case-insensitive checks on the name
        if p.dim() <= 1 or n.endswith("bias") or any(k in n for k in norm_keywords):
            no_decay.append(p)                   # 1D tensors and obvious norms: do NOT decay
        else:
            decay.append(p)                      # Everything else: typically weight matrices/filters

    return decay, no_decay                       # Return the two groups for the optimizer


def build_optimizer(
    model: torch.nn.Module,
    lr: float = 2e-4,
    wd: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    fused: bool = None,
) -> torch.optim.Optimizer:
    """
    Create an AdamW optimizer with two parameter groups (decay / no-decay).
    We optionally enable PyTorch's fused AdamW when available (faster on CUDA builds).

    Args:
        model:  the nn.Module whose parameters we will update
        lr:     learning rate (global scale on all groups)
        wd:     weight decay for the 'decay' group; 'no_decay' group uses 0
        betas:  AdamW momentum coefficients (β₁, β₂)
        eps:    AdamW numerical epsilon (stability in low‑variance regimes)
        fused:  if True/False, force fused AdamW on/off; if None, auto‑detect support

    Returns:
        torch.optim.Optimizer instance ready to use in training loops.
    """
    decay, no_decay = _split_decay_groups(model)          # Make our two param buckets

    groups = [                                            # Set up the two groups with distinct decay
        {"params": decay, "weight_decay": wd},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    # Prepare optional kwargs for fused AdamW without breaking older PyTorch versions.
    extra: Dict[str, Any] = {"lr": lr, "betas": betas, "eps": eps}
    if fused is not None:
        # Caller forced a choice explicitly.
        extra["fused"] = bool(fused)
    else:
        # Try to enable fused automatically if the build supports it and CUDA is present.
        try:
            import inspect
            if "fused" in inspect.signature(torch.optim.AdamW).parameters and torch.cuda.is_available():
                extra["fused"] = True
        except Exception:
            pass  # If anything goes wrong, we simply don't pass 'fused'.

    opt = torch.optim.AdamW(groups, **extra)             # Construct the optimizer with our groups and kwargs
    return opt                                            # Ready to step()


def clip_gradients(model: torch.nn.Module, max_norm: float = 1.0, norm_type: float = 2.0) -> float:
    """
    Clamp the global gradient norm to keep updates stable.
    This prevents a single bad batch from exploding the optimizer state.

    Args:
        model:     the nn.Module holding gradients we want to clip
        max_norm:  upper bound on the overall gradient norm
        norm_type: which norm to use (2.0 for L2, inf for L∞, etc.)

    Returns:
        The total norm of the parameters (useful for telemetry/logging).
    """
    params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]  # Only trainable with grads
    if not params:                                          # If there are no gradients yet, nothing to do
        return 0.0
    return torch.nn.utils.clip_grad_norm_(params, max_norm, norm_type=norm_type)  # Perform clipping in-place


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_scale: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Learning‑rate schedule with a gentle warmup and cosine decay:

        scale(step) = { linear 0→1 over warmup
                      { cosine 1→min_lr_scale over the rest

    We return a LambdaLR so you can call scheduler.step() each iteration.

    Args:
        optimizer:      the AdamW instance we created above
        warmup_steps:   number of steps to ramp up from 0 to 1×lr
        total_steps:    total number of training steps we expect to run
        min_lr_scale:   the final fraction of lr at the end of training (e.g., 0.1 → 10% of lr)

    Returns:
        A torch.optim.lr_scheduler.LambdaLR scheduler.
    """
    warm = max(1, int(warmup_steps))                       # Avoid division by zero; ensure at least 1 step of warmup
    total = max(warm + 1, int(total_steps))                # Ensure total > warmup so cosine phase exists
    min_s = float(min_lr_scale)                            # Cast to float for math

    def scale(step: int) -> float:
        if step < warm:                                    # Linear ramp from 0 → 1
            return float(step + 1) / float(warm)
        # Cosine from 1 → min_s across (total - warm) steps
        prog = min(1.0, float(step - warm) / float(max(1, total - warm)))
        # cos goes 1→−1; we remap to 1→min_s
        return min_s + 0.5 * (1.0 - min_s) * (1.0 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scale)  # Wrap the schedule in a LambdaLR


def zero_grad(optimizer: torch.optim.Optimizer) -> None:
    """
    Reset gradients in a way that avoids memory churn.
    set_to_none=True tells PyTorch to skip filling tensors with zeros — it simply drops them,
    which is both faster and gentler on the allocator.
    """
    optimizer.zero_grad(set_to_none=True)                  # Recommended idiom for modern PyTorch


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Fetch the current learning rate from the first param group.
    Handy for logging and sanity checks.
    """
    return float(optimizer.param_groups[0]["lr"])          # Param groups share the same lr in our setup


def count_params(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Count parameters for quick model size telemetry.

    Returns:
        (trainable, total) as integer counts.
    """
    total = sum(p.numel() for p in model.parameters())     # Count all parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)  # Only those that will update
    return int(trainable), int(total)                      # Return plain ints for easy printing/logging


def set_requires_grad(module: torch.nn.Module, requires_grad: bool) -> None:
    """
    Freeze or unfreeze an entire module tree (and its parameters).
    Useful for staged training, adapter fine‑tuning, etc.
    """
    for p in module.parameters():                          # Walk all parameters in the subtree
        p.requires_grad = bool(requires_grad)              # Flip the flag according to the caller
