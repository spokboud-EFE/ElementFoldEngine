# ElementFold · quantize.py
# ============================================================
# Quantization helpers for the Fold–Gate–Norm engine.
#
# Purpose:
#   Quantization allows ElementFold to run coherently on
#   low-precision (INT8) hardware without breaking phase laws.
#
# Overview of contents:
#   1. Core constants and safe guards.
#   2. Scale selection and tensor (de)quantization primitives.
#   3. Observers and fake-quantization for training simulation.
#   4. Weight-only quantized layers: QLinear, QConv1d.
#   5. Model-level converters (quantize ↔ float).
#   6. Diagnostics and small ergonomic helpers.
#
# Plain words:
#   We pack numbers into INT8 [−127, +127] using symmetric scales.
#   This keeps the algebra centered (zero → zero) and predictable.
#   The “EMA observers” act like slow relaxation meters: they
#   estimate the needed scale s = max|x| / 127 over time.
#   Weight-only quantization saves memory and stays portable.
# ============================================================

from __future__ import annotations
from typing import Tuple, Dict, Any, Optional, Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 0. Core constants and tiny guards
# ============================================================
# Symmetric range avoids −128 asymmetry and keeps rounding unbiased.

INT8_QMIN_SYM = -127
INT8_QMAX_SYM = +127
EPS = 1e-12  # small epsilon to avoid division by zero


# ============================================================
# 1. Scale selection and tensor (de)quantization
# ============================================================

def _choose_scale_symmetric(x: torch.Tensor,
                            per_channel: bool = False,
                            ch_axis: int = 0) -> torch.Tensor:
    """
    Compute a symmetric scaling factor `s` so that x/s ∈ [−127, +127].

    If per_channel=True, we compute one scale per output channel
    along `ch_axis` — used for weights where each row or filter
    can have its own dynamic range.

    Returns:
        s : scalar tensor (per-tensor) or vector (per-channel)
    """
    with torch.no_grad():
        if per_channel:
            dims = [i for i in range(x.dim()) if i != ch_axis]
            maxabs = x.detach().abs().amax(dim=dims, keepdim=False).to(torch.float32)
        else:
            maxabs = x.detach().abs().max().to(torch.float32)
        s = (maxabs.clamp_min(EPS) / float(INT8_QMAX_SYM))
        return s


def quantize_tensor(x: torch.Tensor,
                    scale: torch.Tensor | None = None,
                    zero_point: int | torch.Tensor = 0,
                    per_channel: bool = False,
                    ch_axis: int = 0,
                    qmin: int = INT8_QMIN_SYM,
                    qmax: int = INT8_QMAX_SYM
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize float tensor x → int8 using symmetric scaling.

    Returns
    -------
    q : int8 tensor (values in [−127,+127])
    s : scale tensor (per-tensor or per-channel)
    z : zero point (always 0 here)
    """
    s = _choose_scale_symmetric(x, per_channel, ch_axis) if scale is None else scale.to(torch.float32)
    z = torch.as_tensor(zero_point, dtype=torch.float32, device=x.device)
    if per_channel:
        shape = [1] * x.dim()
        shape[ch_axis] = -1
        s_view = s.view(shape)
    else:
        s_view = s
    q = torch.round(x / s_view).clamp(qmin, qmax).to(torch.int8)
    return q, s, z.to(x.device)


def dequantize_tensor(q: torch.Tensor,
                      scale: torch.Tensor,
                      zero_point: int | torch.Tensor = 0) -> torch.Tensor:
    """
    Convert INT8 tensor back to float using scale and zero-point.
    Plain words: multiply by the scale — the inverse of quantization.
    """
    z = torch.as_tensor(zero_point, dtype=torch.float32, device=q.device)
    return (q.to(torch.float32) - z) * scale.to(torch.float32)


# ============================================================
# 2. Observers and fake quantization (training simulation)
# ============================================================

class EMAMaxAbsObserver(nn.Module):
    """
    Track an exponential-moving average of max|x| to estimate scale.

    Used to calibrate dynamic ranges during training without
    storing full activation histograms.

    Parameters
    ----------
    momentum : float
        How fast the running value adapts (0.95 → slow, stable).
    per_channel : bool
        Whether to track per-channel or global magnitude.
    ch_axis : int
        Channel axis for per-channel mode.
    """
    def __init__(self, momentum: float = 0.95,
                 per_channel: bool = False, ch_axis: int = 0):
        super().__init__()
        self.momentum = float(momentum)
        self.per_channel = bool(per_channel)
        self.ch_axis = int(ch_axis)
        self.register_buffer("running", None, persistent=False)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Update the running EMA of |x|."""
        if self.per_channel:
            dims = [i for i in range(x.dim()) if i != self.ch_axis]
            maxabs = x.detach().abs().amax(dim=dims, keepdim=False).to(torch.float32)
        else:
            maxabs = x.detach().abs().max().to(torch.float32)
        if self.running is None:
            self.running = maxabs
        else:
            self.running.mul_(self.momentum).add_(maxabs, alpha=(1.0 - self.momentum))

    @torch.no_grad()
    def get_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return current (scale, zero-point).
        s = EMA(max|x|)/127, z=0 in symmetric scheme.
        """
        if self.running is None:
            s = torch.tensor(1.0)
        else:
            s = (self.running.clamp_min(EPS) / float(INT8_QMAX_SYM))
        z = torch.tensor(0.0)
        return s, z


def fake_quantize(x: torch.Tensor,
                  observer: Optional[EMAMaxAbsObserver] = None,
                  per_channel: bool = False,
                  ch_axis: int = -1) -> torch.Tensor:
    """
    Simulate quantization (quantize → dequantize) but keep float tensors.
    This preview lets you see quantization effects during training.

    If observer provided:
        update its running scale and use that value.
    Else:
        compute a fresh scale from this batch.
    """
    if observer is not None:
        observer.update(x)
        s, _ = observer.get_params()
        if per_channel and x.dim() > 0:
            shape = [1] * x.dim()
            shape[ch_axis] = -1
            s = s.view(shape).to(x.device)
    else:
        s = _choose_scale_symmetric(x, per_channel=per_channel, ch_axis=ch_axis)
        if per_channel and x.dim() > 0:
            shape = [1] * x.dim()
            shape[ch_axis] = -1
            s = s.view(shape)
    q = torch.round(x / s).clamp(INT8_QMIN_SYM, INT8_QMAX_SYM).to(torch.int8)
    return q.to(torch.float32) * s


def fake_quantize_ste(x: torch.Tensor,
                      observer: Optional[EMAMaxAbsObserver] = None,
                      per_channel: bool = False,
                      ch_axis: int = -1) -> torch.Tensor:
    """
    Straight-Through Estimator (STE) version of fake-quantization.
    Forward: behaves like fake_quantize().
    Backward: gradient passes as identity (dL/dx ≈ dL/dy).
    """
    with torch.no_grad():
        y_nograd = fake_quantize(x, observer=observer,
                                 per_channel=per_channel, ch_axis=ch_axis)
    # (y_nograd - x).detach() cuts gradient; +x restores it.
    return (y_nograd - x).detach() + x


class QuantStub(nn.Module):
    """
    Small module you can insert before layers to simulate activation quantization.
    It owns an EMA observer that learns the scale automatically.
    """
    def __init__(self, per_channel: bool = False, ch_axis: int = -1,
                 ste: bool = False, momentum: float = 0.95):
        super().__init__()
        self.observer = EMAMaxAbsObserver(momentum=momentum,
                                          per_channel=per_channel, ch_axis=ch_axis)
        self.per_channel = per_channel
        self.ch_axis = ch_axis
        self.ste = bool(ste)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ste and self.training:
            return fake_quantize_ste(x, observer=self.observer,
                                     per_channel=self.per_channel, ch_axis=self.ch_axis)
        return fake_quantize(x, observer=self.observer,
                             per_channel=self.per_channel, ch_axis=self.ch_axis)


class DeQuantStub(nn.Module):
    """Identity layer for compatibility with frameworks that separate Quant/DeQuant."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# ============================================================
# 3. Weight-only quantized layers
# ============================================================

class QLinear(nn.Module):
    """
    Weight-only int8 Linear layer (drop-in for nn.Linear).

    We store:
        weight_q : int8 weights
        scale_w  : per-output scale (float)
        bias     : float bias

    On forward we dequantize W = weight_q * scale_w[:,None]
    and use F.linear(x, W, bias).
    """
    def __init__(self, weight_q: torch.Tensor, scale_w: torch.Tensor,
                 bias: torch.Tensor | None, in_features: int, out_features: int):
        super().__init__()
        self.register_buffer("weight_q", weight_q.to(torch.int8), persistent=True)
        self.register_buffer("scale_w", scale_w.to(torch.float32), persistent=True)
        if bias is not None:
            self.register_buffer("bias", bias.to(torch.float32), persistent=True)
        else:
            self.bias = None
        self.in_features = int(in_features)
        self.out_features = int(out_features)

    @classmethod
    def from_float(cls, m: nn.Linear) -> "QLinear":
        """Quantize a float nn.Linear into QLinear."""
        W = m.weight.detach()
        s = _choose_scale_symmetric(W, per_channel=True, ch_axis=0)
        q = torch.round(W / s.unsqueeze(1)).clamp(INT8_QMIN_SYM, INT8_QMAX_SYM).to(torch.int8)
        b = None if m.bias is None else m.bias.detach()
        return cls(q, s, b, W.size(1), W.size(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.weight_q.to(torch.float32) * self.scale_w.unsqueeze(1)
        return F.linear(x, W, self.bias)


class QConv1d(nn.Module):
    """
    Weight-only int8 Conv1d.
    Stores int8 weights + per-out-channel scale.
    On forward, dequantizes to float and calls F.conv1d.
    """
    def __init__(self, weight_q: torch.Tensor, scale_w: torch.Tensor,
                 bias: torch.Tensor | None, stride: int | Tuple[int] = 1,
                 padding: int | Tuple[int] = 0, dilation: int | Tuple[int] = 1,
                 groups: int = 1):
        super().__init__()
        self.register_buffer("weight_q", weight_q.to(torch.int8), persistent=True)
        self.register_buffer("scale_w", scale_w.to(torch.float32), persistent=True)
        if bias is not None:
            self.register_buffer("bias", bias.to(torch.float32), persistent=True)
        else:
            self.bias = None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = int(groups)

    @classmethod
    def from_float(cls, m: nn.Conv1d) -> "QConv1d":
        W = m.weight.detach()
        s = _choose_scale_symmetric(W, per_channel=True, ch_axis=0)
        q = torch.round(W / s.view(-1, 1, 1)).clamp(INT8_QMIN_SYM, INT8_QMAX_SYM).to(torch.int8)
        b = None if m.bias is None else m.bias.detach()
        return cls(q, s, b, m.stride, m.padding, m.dilation, m.groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.weight_q.to(torch.float32) * self.scale_w.view(-1, 1, 1)
        return F.conv1d(x, W, self.bias,
                        stride=self.stride, padding=self.padding,
                        dilation=self.dilation, groups=self.groups)


# ============================================================
# 4. Model-level converters
# ============================================================

def quantize_module(m: nn.Module) -> nn.Module:
    """Replace a float module with its quantized version if supported."""
    if isinstance(m, nn.Linear):
        return QLinear.from_float(m)
    if isinstance(m, nn.Conv1d):
        return QConv1d.from_float(m)
    return m


def to_float_module(m: nn.Module) -> nn.Module:
    """Inverse: convert quantized QLinear/QConv1d back to float modules."""
    if isinstance(m, QLinear):
        device = m.weight_q.device
        dtype = torch.float32
        mm = nn.Linear(m.in_features, m.out_features,
                       bias=(m.bias is not None)).to(device)
        with torch.no_grad():
            W = m.weight_q.to(dtype) * m.scale_w.unsqueeze(1)
            mm.weight.copy_(W)
            if m.bias is not None:
                mm.bias.copy_(m.bias.to(dtype))
        return mm

    if isinstance(m, QConv1d):
        device = m.weight_q.device
        dtype = torch.float32
        O, I_over_G, K = m.weight_q.size()
        G = int(m.groups)
        mm = nn.Conv1d(in_channels=I_over_G * G, out_channels=O, kernel_size=K,
                       stride=m.stride, padding=m.padding, dilation=m.dilation,
                       groups=G, bias=(m.bias is not None)).to(device)
        with torch.no_grad():
            W = m.weight_q.to(dtype) * m.scale_w.view(-1, 1, 1)
            mm.weight.copy_(W)
            if m.bias is not None:
                mm.bias.copy_(m.bias.to(dtype))
        return mm

    return m


def quantize_model_weights(model: nn.Module) -> nn.Module:
    """Recursively replace supported submodules with weight-only int8 versions."""
    for name, child in list(model.named_children()):
        qchild = quantize_module(child)
        setattr(model, name, qchild)
        quantize_model_weights(qchild)
    return model


def dequantize_model_weights(model: nn.Module) -> nn.Module:
    """Recursively convert quantized modules back to float versions."""
    for name, child in list(model.named_children()):
        fchild = to_float_module(child)
        setattr(model, name, fchild)
        dequantize_model_weights(fchild)
    return model


# ============================================================
# 5. Diagnostics & ergonomics
# ============================================================

def count_int8_params(model: nn.Module) -> Tuple[int, int]:
    """
    Count how many parameters/buffers are stored as int8.
    Returns (int8_params, total_params).
    """
    total = 0
    int8_params = 0
    for p in model.parameters(recurse=True):
        total += p.numel()
    for b in model.buffers(recurse=True):
        total += b.numel()
        if b.dtype == torch.int8 and not b.requires_grad:
            int8_params += b.numel()
    return int(int8_params), int(total)


def weight_error_report(m: nn.Module) -> Dict[str, float]:
    """
    Quick reconstruction error report for quantized weights.
    (Compares dequantized weights to their stored quantized form.)
    """
    if isinstance(m, QLinear):
        Wq = m.weight_q.to(torch.float32) * m.scale_w.unsqueeze(1)
        return {
            "max_abs_diff": float((Wq - Wq.detach()).abs().max().item()),
            "mean_abs_diff": float((Wq - Wq.detach()).abs().mean().item())
        }
    if isinstance(m, QConv1d):
        Wq = m.weight_q.to(torch.float32) * m.scale_w.view(-1, 1, 1)
        return {
            "max_abs_diff": float((Wq - Wq.detach()).abs().max().item()),
            "mean_abs_diff": float((Wq - Wq.detach()).abs().mean().item())
        }
    return {}


# ============================================================
# 6. Friendly aliases and exports
# ============================================================

def quantize_tensor_symmetric(x: torch.Tensor,
                              per_channel: bool = False,
                              ch_axis: int = 0):
    """Shortcut for symmetric int8 quantization with z=0."""
    return quantize_tensor(x, per_channel=per_channel, ch_axis=ch_axis,
                           qmin=INT8_QMIN_SYM, qmax=INT8_QMAX_SYM)


__all__ = [
    "EMAMaxAbsObserver", "fake_quantize", "fake_quantize_ste",
    "QuantStub", "DeQuantStub",
    "quantize_tensor", "dequantize_tensor", "quantize_tensor_symmetric",
    "QLinear", "QConv1d",
    "quantize_module", "to_float_module",
    "quantize_model_weights", "dequantize_model_weights",
    "count_int8_params", "weight_error_report",
    "INT8_QMIN_SYM", "INT8_QMAX_SYM",
]
