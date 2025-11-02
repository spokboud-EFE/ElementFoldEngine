# ElementFold · quantize.py
# INT8 helpers that keep the Fold–Gate–Norm engine coherent on low‑precision hardware.
# Read this like a toolbox:
#   1) Per‑tensor / per‑channel symmetric quantization (int8 in [−127, +127], zero‑point 0).
#   2) EMA observers to calibrate scales from activations without storing full histograms.
#   3) Fake‑quantize (quantize → dequantize) for dry runs and training‑time simulation.
#   4) Weight‑only quantized modules: QLinear and QConv1d (drop‑in for nn.Linear / nn.Conv1d).
#   5) Whole‑model converters that swap modules in place for inference.

from __future__ import annotations                                    # ↻ future annotations on older Python
from typing import Tuple, Dict, Any, Iterable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ———————————————————————————————————————————————————————————————————————————
# 0) Core constants and tiny guards
# ———————————————————————————————————————————————————————————————————————————

INT8_QMIN_SYM = -127                                                   # symmetric int8 range we target
INT8_QMAX_SYM = +127                                                   # we avoid −128 to keep symmetry exact
EPS = 1e-12                                                            # tiny epsilon to avoid divide‑by‑zero


# ———————————————————————————————————————————————————————————————————————————
# 1) Scale selection and (de)quantization primitives
# ———————————————————————————————————————————————————————————————————————————

def _choose_scale_symmetric(x: torch.Tensor, per_channel: bool = False, ch_axis: int = 0) -> torch.Tensor:
    """
    Pick a symmetric scale `s` so that `x/s` fits into [−127, +127].
    If per_channel=True, compute one scale per channel along `ch_axis`.
    """
    with torch.no_grad():
        if per_channel:
            # Reduce all dimensions except the channel axis → per‑channel max |x|
            dims = [i for i in range(x.dim()) if i != ch_axis]
            maxabs = x.detach().abs().amax(dim=dims, keepdim=False)    # shape: (C,)
        else:
            maxabs = x.detach().abs().max()                            # scalar
        s = (maxabs.clamp_min(EPS) / float(INT8_QMAX_SYM)).to(torch.float32)
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
        z  — zero‑point tensor (here always 0 for symmetric)
    """
    # 1) Choose or broadcast scale
    s = _choose_scale_symmetric(x, per_channel, ch_axis) if scale is None else scale.to(torch.float32)
    # 2) Normalize and round; zero‑point z=0 is implied by symmetry
    z = torch.as_tensor(zero_point, dtype=torch.float32, device=x.device)
    if per_channel:
        # Unsqueeze scale along non‑channel dims to match x’s shape for broadcasting
        shape = [1] * x.dim()
        shape[ch_axis] = -1
        s_ = s.view(shape)
    else:
        s_ = s
    q = torch.round(x / s_).clamp(qmin, qmax).to(torch.int8)           # int8 tensor
    return q, s, z.to(x.device)


def dequantize_tensor(q: torch.Tensor, scale: torch.Tensor, zero_point: int | torch.Tensor = 0) -> torch.Tensor:
    """
    Convert int8 tensor back to float using the stored scale (and z=0).
    """
    z = torch.as_tensor(zero_point, dtype=torch.float32, device=q.device)
    return (q.to(torch.float32) - z) * scale.to(torch.float32)


# ———————————————————————————————————————————————————————————————————————————
# 2) Observers and fake quantization (training‑time simulation)
# ———————————————————————————————————————————————————————————————————————————

class EMAMaxAbsObserver(nn.Module):
    """
    Track an exponential‑moving average of max|x| to calibrate symmetric scales.
    Useful for dynamic activations (one scale per tensor or per channel).
    """
    def __init__(self, momentum: float = 0.95, per_channel: bool = False, ch_axis: int = 0):
        super().__init__()
        self.momentum = float(momentum)
        self.per_channel = bool(per_channel)
        self.ch_axis = int(ch_axis)
        self.register_buffer("running", None, persistent=False)         # holds scalar or (C,) tensor

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
    This is handy during training to preview quantization effects.

    If an observer is provided, we update it and use its EMA scale; otherwise we
    compute a fresh scale from the current batch.
    """
    if observer is not None:
        observer.update(x)
        s, z = observer.get_params()
        if per_channel and x.dim() > 0:
            # Align channel axis: observer stores channel along ch_axis; we broadcast accordingly
            shape = [1] * x.dim()
            shape[ch_axis] = -1
            s = s.view(shape).to(x.device)
    else:
        s = _choose_scale_symmetric(x, per_channel=per_channel, ch_axis=ch_axis)
        z = torch.tensor(0.0, device=x.device)
        if per_channel and x.dim() > 0:
            shape = [1] * x.dim()
            shape[ch_axis] = -1
            s = s.view(shape)

    q = torch.round(x / s).clamp(INT8_QMIN_SYM, INT8_QMAX_SYM).to(torch.int8)
    return q.to(torch.float32) * s                                                    # dequantized float (fake‑quant)


# ———————————————————————————————————————————————————————————————————————————
# 3) Weight‑only quantized layers (drop‑in for nn.Linear / nn.Conv1d)
# ———————————————————————————————————————————————————————————————————————————

class QLinear(nn.Module):
    """
    Weight‑only int8 linear layer.
    We store W_q ∈ int8 and scale per OUT channel; on forward we dequantize to float and run F.linear.
    (Real backends can run int8×int8 GEMM; we keep this portable and dependency‑free.)
    """
    def __init__(self, weight_q: torch.Tensor, scale_w: torch.Tensor, bias: torch.Tensor | None, in_features: int, out_features: int):
        super().__init__()
        self.register_buffer("weight_q", weight_q.to(torch.int8), persistent=True)    # (O, I) int8
        self.register_buffer("scale_w", scale_w.to(torch.float32), persistent=True)   # (O,) float
        if bias is not None:
            self.register_buffer("bias", bias.to(torch.float32), persistent=True)     # (O,) float
        else:
            self.bias = None
        self.in_features = int(in_features)
        self.out_features = int(out_features)

    @classmethod
    def from_float(cls, m: nn.Linear) -> "QLinear":
        W = m.weight.detach()                                                         # (O, I) float
        # Per‑OUT‑channel scaling (row‑wise for nn.Linear weights)
        s = _choose_scale_symmetric(W, per_channel=True, ch_axis=0)                   # (O,)
        q = torch.round(W / s.unsqueeze(1)).clamp(INT8_QMIN_SYM, INT8_QMAX_SYM).to(torch.int8)
        b = None if m.bias is None else m.bias.detach()
        return cls(q, s, b, W.size(1), W.size(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.weight_q.to(torch.float32) * self.scale_w.unsqueeze(1)               # dequantize rows
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
        self.register_buffer("weight_q", weight_q.to(torch.int8), persistent=True)    # (O, I/G, K) int8
        self.register_buffer("scale_w", scale_w.to(torch.float32), persistent=True)   # (O,) float
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
        W = m.weight.detach()                                                         # (O, I/G, K)
        s = _choose_scale_symmetric(W, per_channel=True, ch_axis=0)                   # (O,)
        q = torch.round(W / s.view(-1, 1, 1)).clamp(INT8_QMIN_SYM, INT8_QMAX_SYM).to(torch.int8)
        b = None if m.bias is None else m.bias.detach()
        return cls(q, s, b, m.stride, m.padding, m.dilation, m.groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.weight_q.to(torch.float32) * self.scale_w.view(-1, 1, 1)             # dequantize per OUT
        return F.conv1d(x, W, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)


# ———————————————————————————————————————————————————————————————————————————
# 4) Model‑level converters
# ———————————————————————————————————————————————————————————————————————————

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


def quantize_model_weights(model: nn.Module) -> nn.Module:
    """
    Walk the module tree and replace supported layers with weight‑only int8 versions.
    This is safe for inference: shapes and outputs remain compatible (modulo tiny quantization error).
    """
    for name, child in list(model.named_children()):
        qchild = quantize_module(child)                          # Convert this layer if supported
        setattr(model, name, qchild)                             # Install back
        quantize_model_weights(qchild)                           # Recurse into children
    return model


# ———————————————————————————————————————————————————————————————————————————
# 5) Public API (symmetric tensor quantization helpers)
# ———————————————————————————————————————————————————————————————————————————

def quantize_tensor_symmetric(x: torch.Tensor, per_channel: bool = False, ch_axis: int = 0):
    """
    Friendly alias for symmetric int8 quantization with zero‑point 0.
    """
    return quantize_tensor(x, per_channel=per_channel, ch_axis=ch_axis, qmin=INT8_QMIN_SYM, qmax=INT8_QMAX_SYM)


__all__ = [
    "EMAMaxAbsObserver",
    "fake_quantize",
    "quantize_tensor",
    "dequantize_tensor",
    "quantize_tensor_symmetric",
    "QLinear",
    "QConv1d",
    "quantize_module",
    "quantize_model_weights",
    "INT8_QMIN_SYM",
    "INT8_QMAX_SYM",
]
