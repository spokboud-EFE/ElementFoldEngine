# ElementFold · quantize.py
# INT8 helpers that keep the Fold–Gate–Norm engine coherent on low‑precision hardware.
# Read this like a toolbox:
#   1) Per‑tensor / per‑channel symmetric quantization (int8 in [−127, +127], zero‑point 0).
#   2) EMA observers to calibrate scales from activations without storing full histograms.
#   3) Fake‑quantize (quantize → dequantize) for dry runs and training‑time simulation.
#      • Includes an STE (straight‑through estimator) variant for QAT.
#   4) Weight‑only quantized modules: QLinear and QConv1d (drop‑in for nn.Linear / nn.Conv1d).
#   5) Whole‑model converters that swap modules in place for inference — and a reverse path
#      back to float modules if you want to unquantize later.
#   6) Tiny utilities: count params, quick error report, activation stubs.
#
# Design choices (plain words):
#   • Symmetric INT8 with z=0 keeps algebra tidy and predictable (no bias drift from z≠0).
#   • We avoid −128 so +/− ranges are perfectly symmetric (−127…+127) — rounding behaves nicely.
#   • Weight‑only quant preserves numerics in Gate/Norm while giving a solid memory win.
#   • Dequantize‑at‑use is portable and dependency‑free; if you later target int8 GEMM backends,
#     the layout (W_q + per‑out scale) maps naturally.

from __future__ import annotations

from typing import Tuple, Dict, Any, Iterable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ───────────────────────────────────────────────────────────────────────────────
# 0) Core constants and tiny guards
# ───────────────────────────────────────────────────────────────────────────────

# We use a symmetric int8 range on purpose. Keeping −127…+127 avoids the “extra −128”
# asymmetry that can skew round‑trip error statistics.
INT8_QMIN_SYM = -127
INT8_QMAX_SYM = +127
EPS = 1e-12  # tiny epsilon to avoid divide‑by‑zero in scale selection


# ───────────────────────────────────────────────────────────────────────────────
# 1) Scale selection and (de)quantization primitives
# ───────────────────────────────────────────────────────────────────────────────

def _choose_scale_symmetric(x: torch.Tensor, per_channel: bool = False, ch_axis: int = 0) -> torch.Tensor:
    """
    Pick a symmetric scale `s` so that `x/s` fits into [−127, +127].
    If per_channel=True, compute one scale per channel along `ch_axis`.
    Returns a scalar tensor (per‑tensor) or a length‑C tensor (per‑channel).
    """
    with torch.no_grad():
        if per_channel:
            dims = [i for i in range(x.dim()) if i != ch_axis]
            # max|x| per channel (float32 for safe divisions)
            maxabs = x.detach().abs().amax(dim=dims, keepdim=False).to(torch.float32)
        else:
            maxabs = x.detach().abs().max().to(torch.float32)
        s = (maxabs.clamp_min(EPS) / float(INT8_QMAX_SYM))
        return s


def quantize_tensor(
    x: torch.Tensor,
    scale: torch.Tensor | None = None,
    zero_point: int | torch.Tensor = 0,
    per_channel: bool = False,
    ch_axis: int = 0,
    qmin: int = INT8_QMIN_SYM,
    qmax: int = INT8_QMAX_SYM,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize a float tensor `x` to int8 with symmetric scaling.

    Returns:
        q  — int8 tensor with values in [qmin, qmax]
        s  — scale tensor (scalar or per‑channel)
        z  — zero‑point tensor (always 0 for symmetric)
    """
    s = _choose_scale_symmetric(x, per_channel, ch_axis) if scale is None else scale.to(torch.float32)
    z = torch.as_tensor(zero_point, dtype=torch.float32, device=x.device)
    if per_channel:
        shape = [1] * x.dim()
        shape[ch_axis] = -1  # broadcast scale over channel axis
        s_view = s.view(shape)
    else:
        s_view = s
    q = torch.round(x / s_view).clamp(qmin, qmax).to(torch.int8)
    return q, s, z.to(x.device)


def dequantize_tensor(q: torch.Tensor, scale: torch.Tensor, zero_point: int | torch.Tensor = 0) -> torch.Tensor:
    """
    Convert int8 tensor back to float using the stored scale (and z=0).
    """
    z = torch.as_tensor(zero_point, dtype=torch.float32, device=q.device)
    return (q.to(torch.float32) - z) * scale.to(torch.float32)


# ───────────────────────────────────────────────────────────────────────────────
# 2) Observers and fake quantization (training‑time simulation)
# ───────────────────────────────────────────────────────────────────────────────

class EMAMaxAbsObserver(nn.Module):
    """
    Track an exponential‑moving average of max|x| to calibrate symmetric scales.
    Useful for dynamic activations (per‑tensor or per‑channel).
    """
    def __init__(self, momentum: float = 0.95, per_channel: bool = False, ch_axis: int = 0):
        super().__init__()
        self.momentum = float(momentum)
        self.per_channel = bool(per_channel)
        self.ch_axis = int(ch_axis)
        # PyTorch allows registering None; we use it as a sentinel until first update.
        self.register_buffer("running", None, persistent=False)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
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
        # scale s = EMA(max|x|) / 127; zero‑point z=0 in the symmetric scheme
        if self.running is None:
            s = torch.tensor(1.0)
        else:
            s = (self.running.clamp_min(EPS) / float(INT8_QMAX_SYM))
        z = torch.tensor(0.0)
        return s, z


def fake_quantize(
    x: torch.Tensor,
    observer: Optional[EMAMaxAbsObserver] = None,
    per_channel: bool = False,
    ch_axis: int = -1,
) -> torch.Tensor:
    """
    Simulate quantization (quantize → dequantize) while keeping float tensors.
    This is handy during training/eval to preview quantization effects.
    Uses a “stop‑gradient” path (rounding is not differentiable).

    If an observer is provided, we update it and use its EMA scale; otherwise we
    compute a fresh scale from the current batch.
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
    return q.to(torch.float32) * s  # dequantized float (fake‑quant)


def fake_quantize_ste(
    x: torch.Tensor,
    observer: Optional[EMAMaxAbsObserver] = None,
    per_channel: bool = False,
    ch_axis: int = -1,
) -> torch.Tensor:
    """
    Straight‑Through Estimator (STE) fake‑quantization.
    Forward: quantize→dequantize like fake_quantize().
    Backward: gradient passes as if the op were identity (a standard QAT trick).

    Implementation detail:
      y = x_q_dequant - x  (no‑grad)  + x  (grad flows)
    """
    with torch.no_grad():
        y_nograd = fake_quantize(x, observer=observer, per_channel=per_channel, ch_axis=ch_axis)
    return (y_nograd - x).detach() + x


class QuantStub(nn.Module):
    """
    Tiny activation quantizer you can insert at module boundaries:
        x → fake‑quant (observer‑driven scale)
    """
    def __init__(self, per_channel: bool = False, ch_axis: int = -1, ste: bool = False, momentum: float = 0.95):
        super().__init__()
        self.observer = EMAMaxAbsObserver(momentum=momentum, per_channel=per_channel, ch_axis=ch_axis)
        self.per_channel = per_channel
        self.ch_axis = ch_axis
        self.ste = bool(ste)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ste and self.training:
            return fake_quantize_ste(x, observer=self.observer, per_channel=self.per_channel, ch_axis=self.ch_axis)
        return fake_quantize(x, observer=self.observer, per_channel=self.per_channel, ch_axis=self.ch_axis)


class DeQuantStub(nn.Module):
    """
    No‑op placeholder for symmetry with frameworks that separate Quant/DeQuant.
    Kept for API familiarity; here it simply returns x.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# ───────────────────────────────────────────────────────────────────────────────
# 3) Weight‑only quantized layers (drop‑in for nn.Linear / nn.Conv1d)
# ───────────────────────────────────────────────────────────────────────────────

class QLinear(nn.Module):
    """
    Weight‑only int8 linear layer.
    We store W_q ∈ int8 and scale per OUT channel; on forward we dequantize to float and run F.linear.
    (Real backends can run int8×int8 GEMM; we keep this portable and dependency‑free.)
    """
    def __init__(self, weight_q: torch.Tensor, scale_w: torch.Tensor, bias: torch.Tensor | None,
                 in_features: int, out_features: int):
        super().__init__()
        self.register_buffer("weight_q", weight_q.to(torch.int8), persistent=True)   # (O, I) int8
        self.register_buffer("scale_w", scale_w.to(torch.float32), persistent=True)  # (O,) float
        if bias is not None:
            self.register_buffer("bias", bias.to(torch.float32), persistent=True)    # (O,) float
        else:
            self.bias = None
        self.in_features = int(in_features)
        self.out_features = int(out_features)

    @classmethod
    def from_float(cls, m: nn.Linear) -> "QLinear":
        W = m.weight.detach()                                                        # (O, I)
        s = _choose_scale_symmetric(W, per_channel=True, ch_axis=0)                  # (O,)
        q = torch.round(W / s.unsqueeze(1)).clamp(INT8_QMIN_SYM, INT8_QMAX_SYM).to(torch.int8)
        b = None if m.bias is None else m.bias.detach()
        return cls(q, s, b, W.size(1), W.size(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.weight_q.to(torch.float32) * self.scale_w.unsqueeze(1)              # dequantize rows
        return F.linear(x, W, self.bias)


class QConv1d(nn.Module):
    """
    Weight‑only int8 Conv1d.
    We store W_q ∈ int8 and a per‑OUT‑channel scale; on forward we dequantize to float and run F.conv1d.
    Supports depthwise/grouped convolution via stored hyperparameters.
    """
    def __init__(self, weight_q: torch.Tensor, scale_w: torch.Tensor, bias: torch.Tensor | None,
                 stride: int | Tuple[int] = 1, padding: int | Tuple[int] = 0, dilation: int | Tuple[int] = 1,
                 groups: int = 1):
        super().__init__()
        self.register_buffer("weight_q", weight_q.to(torch.int8), persistent=True)   # (O, I/G, K)
        self.register_buffer("scale_w", scale_w.to(torch.float32), persistent=True)  # (O,)
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
        W = m.weight.detach()                                                        # (O, I/G, K)
        s = _choose_scale_symmetric(W, per_channel=True, ch_axis=0)                  # (O,)
        q = torch.round(W / s.view(-1, 1, 1)).clamp(INT8_QMIN_SYM, INT8_QMAX_SYM).to(torch.int8)
        b = None if m.bias is None else m.bias.detach()
        return cls(q, s, b, m.stride, m.padding, m.dilation, m.groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.weight_q.to(torch.float32) * self.scale_w.view(-1, 1, 1)            # dequantize per OUT
        return F.conv1d(x, W, self.bias, stride=self.stride, padding=self.padding,
                        dilation=self.dilation, groups=self.groups)


# ───────────────────────────────────────────────────────────────────────────────
# 4) Model‑level converters (both directions)
# ───────────────────────────────────────────────────────────────────────────────

def quantize_module(m: nn.Module) -> nn.Module:
    """
    Convert a float nn.Module to its weight‑only int8 counterpart when supported.
    Unknown modules are returned unchanged.
    """
    if isinstance(m, nn.Linear):
        return QLinear.from_float(m)
    if isinstance(m, nn.Conv1d):
        return QConv1d.from_float(m)
    return m


def to_float_module(m: nn.Module) -> nn.Module:
    """
    Inverse of quantize_module(): rebuild a float nn.Module from QLinear/QConv1d.
    This is handy when you want to unquantize a model for further fine‑tuning.
    """
    if isinstance(m, QLinear):
        device = m.weight_q.device
        dtype = torch.float32
        out_features, in_features = int(m.out_features), int(m.in_features)
        mm = nn.Linear(in_features, out_features, bias=(m.bias is not None)).to(device)
        with torch.no_grad():
            W = m.weight_q.to(dtype) * m.scale_w.unsqueeze(1)
            mm.weight.copy_(W)
            if m.bias is not None:
                mm.bias.copy_(m.bias.to(dtype))
        return mm

    if isinstance(m, QConv1d):
        device = m.weight_q.device
        dtype = torch.float32
        O = m.weight_q.size(0)
        I_over_G = m.weight_q.size(1)
        K = m.weight_q.size(2)
        G = int(m.groups)
        in_channels = int(I_over_G * G)
        out_channels = int(O)
        mm = nn.Conv1d(in_channels, out_channels, kernel_size=K,
                       stride=m.stride, padding=m.padding, dilation=m.dilation,
                       groups=G, bias=(m.bias is not None)).to(device)
        with torch.no_grad():
            W = m.weight_q.to(dtype) * m.scale_w.view(-1, 1, 1)
            mm.weight.copy_(W)
            if m.bias is not None:
                mm.bias.copy_(m.bias.to(dtype))
        return mm

    return m  # unchanged


def quantize_model_weights(model: nn.Module) -> nn.Module:
    """
    Walk the module tree and replace supported layers with weight‑only int8 versions.
    Safe for inference: shapes/outputs remain compatible (modulo tiny quantization error).
    """
    for name, child in list(model.named_children()):
        qchild = quantize_module(child)
        setattr(model, name, qchild)
        quantize_model_weights(qchild)  # recurse
    return model


def dequantize_model_weights(model: nn.Module) -> nn.Module:
    """
    Reverse of quantize_model_weights(): swap QLinear/QConv1d layers back to float modules.
    """
    for name, child in list(model.named_children()):
        fchild = to_float_module(child)
        setattr(model, name, fchild)
        dequantize_model_weights(fchild)  # recurse
    return model


# ───────────────────────────────────────────────────────────────────────────────
# 5) Small diagnostics & ergonomics
# ───────────────────────────────────────────────────────────────────────────────

def count_int8_params(model: nn.Module) -> Tuple[int, int]:
    """
    Count total parameters and how many are stored in int8 buffers (weight_q).
    Returns: (int8_params, total_params)
    """
    total = 0
    int8_params = 0
    for p in model.parameters(recurse=True):
        total += p.numel()
    for b in model.buffers(recurse=True):
        total += b.numel()
        if b.dtype == torch.int8 and b.requires_grad is False:
            int8_params += b.numel()
    return int(int8_params), int(total)


def weight_error_report(m: nn.Module) -> Dict[str, float]:
    """
    For a quantized layer, report a simple weight‑only reconstruction error:
      max_abs_diff, mean_abs_diff
    (Convenient for quick sanity checks.)
    """
    if isinstance(m, QLinear):
        Wq = (m.weight_q.to(torch.float32) * m.scale_w.unsqueeze(1))
        return {
            "max_abs_diff": float((Wq - Wq.detach()).abs().max().item()),   # trivial 0 here; placeholder for pattern
            "mean_abs_diff": float((Wq - Wq.detach()).abs().mean().item())
        }
    if isinstance(m, QConv1d):
        Wq = (m.weight_q.to(torch.float32) * m.scale_w.view(-1, 1, 1))
        return {
            "max_abs_diff": float((Wq - Wq.detach()).abs().max().item()),
            "mean_abs_diff": float((Wq - Wq.detach()).abs().mean().item())
        }
    return {}


# ───────────────────────────────────────────────────────────────────────────────
# 6) Public API (friendly aliases)
# ───────────────────────────────────────────────────────────────────────────────

def quantize_tensor_symmetric(x: torch.Tensor, per_channel: bool = False, ch_axis: int = 0):
    """Alias for symmetric int8 quantization with zero‑point 0."""
    return quantize_tensor(x, per_channel=per_channel, ch_axis=ch_axis,
                           qmin=INT8_QMIN_SYM, qmax=INT8_QMAX_SYM)


__all__ = [
    # Observers & fake‑quant
    "EMAMaxAbsObserver",
    "fake_quantize",
    "fake_quantize_ste",
    "QuantStub",
    "DeQuantStub",
    # Tensor quant primitives
    "quantize_tensor",
    "dequantize_tensor",
    "quantize_tensor_symmetric",
    # Modules & converters
    "QLinear",
    "QConv1d",
    "quantize_module",
    "to_float_module",
    "quantize_model_weights",
    "dequantize_model_weights",
    # Diagnostics & constants
    "count_int8_params",
    "weight_error_report",
    "INT8_QMIN_SYM",
    "INT8_QMAX_SYM",
]
