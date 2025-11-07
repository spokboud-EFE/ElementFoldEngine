# ElementFold · align.py
# ============================================================
# Alignment head — teaches representations to stay phase‑aligned
# on the δ★ (delta‑star) circle, without a temperature parameter.
#
# Conceptual picture:
#   • Each sample x lives on a circular ledger measured in δ★ units.
#   • "Positive" examples are tiny, legal micro‑shifts within one click.
#   • "Negative" examples are half‑click offsets (±δ★/2), where similarity
#     should be minimal — the opposite seat on the circle.
#
# Kernel:
#   κ(x,y) = cos[2π(x−y)/δ★]  (exact periodic similarity)
#
# Loss (temperature‑free, δ★ sets the scale):
#   L = − E_b [ log Σ_pos e^{κ⁺} − log Σ_neg e^{κ⁻} ]
#
# Implementation notes:
#   • We DETACH targets (g_pos, g_neg) from x to avoid the zero‑gradient trap
#     when y is constructed as (x + offset). This preserves non‑trivial gradients.
#   • Fully vectorized; robust capacity handling; safe on half/bfloat via the
#     kernel’s internal float32 math.
# ============================================================

from __future__ import annotations
import torch
import torch.nn as nn

# κ(x,y) = cos(2π(x−y)/δ★); defined in core/ledger.py
from .core.ledger import char_kernel


__all__ = ["AlignHead"]


def _expand_caps(capacities, B: int, device: torch.device) -> torch.Tensor:
    """
    Normalize 'capacities' into a (B,) int64 tensor on the given device.
    Falls back to max(C) if shape is incompatible.
    """
    caps = capacities if torch.is_tensor(capacities) else torch.as_tensor(capacities)
    caps = caps.to(device=device)
    if caps.numel() == 1:
        caps_b = caps.view(1).expand(B)
    elif caps.numel() == B:
        caps_b = caps
    else:
        caps_b = caps.max().view(1).expand(B)
    return caps_b.clamp_min(1).to(torch.int64)


class AlignHead(nn.Module):
    """
    Temperature‑free circular contrastive head.

    Parameters
    ----------
    delta : float
        The fixed click size δ★ (sets the periodicity of the circle).

    Forward
    -------
    loss, pos_score, neg_score = align(
        x, capacities,
        num_pos=2, num_neg=8,
        noise=0.02, neg_jitter=0.20
    )

    Inputs
    ------
    x : Tensor (B,) or (B,T)
        Ledger positions. If (B,T), averaged across T to one phase per sample.
    capacities : scalar, (B,), or broadcastable
        Per-sample seat capacities (integer counts C). If the shape
        doesn’t match {1,B}, we fall back to max(C) across the input.
    num_pos : int
        Number of positive micro‑shifts per sample (small, ≈2).
    num_neg : int
        Number of half‑click negatives per sample.
    noise : float
        Tiny additive noise in ledger units to break symmetry.
    neg_jitter : float
        Standard deviation (as fraction of δ★) for half‑click jitter.

    Returns
    -------
    loss : scalar Tensor
        Contrastive loss (lower is better).
    pos_score : float
        Average exp(κ) for positives (alignment quality).
    neg_score : float
        Average exp(κ) for negatives (should be small).
    """

    def __init__(self, delta: float):
        super().__init__()
        self.delta = float(delta)

    def forward(
        self,
        x: torch.Tensor,
        capacities,
        num_pos: int = 2,
        num_neg: int = 8,
        noise: float = 0.02,
        neg_jitter: float = 0.20,
    ):
        # ------------------------------------------------------------
        # 1) Reduce sequences: (B,T) → (B,)
        # ------------------------------------------------------------
        if x.dim() > 1:
            x = x.mean(dim=1)
        B = int(x.shape[0])
        device, dtype = x.device, x.dtype

        # ------------------------------------------------------------
        # 2) Capacity handling and per‑sample seat step Δ_b = δ★ / C_b
        # ------------------------------------------------------------
        caps_b = _expand_caps(capacities, B, device)
        delta = self.delta
        delta_t = torch.as_tensor(delta, device=device, dtype=torch.get_default_dtype())
        step_b = (delta_t / caps_b.to(delta_t.dtype))  # (B,)

        # ------------------------------------------------------------
        # 3) POSITIVES — small micro‑shifts within one click (detach targets)
        # ------------------------------------------------------------
        P = max(1, int(num_pos))
        # Restrict offsets to roughly ±δ★/6 to stay in‑seat
        M_b = torch.clamp(caps_b // 6, min=1)  # (B,)

        # Random integer step m∈[1,M_b] and sign s∈{−1,+1}
        u = torch.rand(P, B, device=device)
        m = torch.floor(u * M_b.unsqueeze(0).to(u.dtype)).to(torch.int64) + 1
        s_pos = (torch.randint(0, 2, (P, B), device=device, dtype=torch.int64) * 2 - 1)

        # Targets are STOP‑GRAD snapshots of x to avoid zero‑grad pathology
        x_anchor = x.to(step_b.dtype)                 # (B,)
        x_targ   = x_anchor.detach()                  # (B,) no grad
        g_pos = (
            x_targ.unsqueeze(0)
            + (s_pos * m).to(step_b.dtype) * step_b.unsqueeze(0)
            + float(noise) * torch.randn(P, B, device=device, dtype=step_b.dtype)
        )

        # ------------------------------------------------------------
        # 4) NEGATIVES — around half‑click (±δ★/2), with jitter (detach targets)
        # ------------------------------------------------------------
        N = max(1, int(num_neg))
        s_neg = (torch.randint(0, 2, (N, B), device=device, dtype=torch.int64) * 2 - 1).to(step_b.dtype)
        half = 0.5 * delta_t
        neg_sigma = float(neg_jitter) * delta
        g_neg = (
            x_targ.unsqueeze(0)
            + s_neg * half
            + neg_sigma * torch.randn(N, B, device=device, dtype=step_b.dtype)
        )

        # ------------------------------------------------------------
        # 5) Similarity on the δ★ circle (kernel does stable float32 internally)
        # ------------------------------------------------------------
        # Shapes: x[None,:] → (1,B); g_pos → (P,B); g_neg → (N,B)
        sim_pos = char_kernel(x_anchor.unsqueeze(0), g_pos, delta)  # (P,B) in [−1,1]
        sim_neg = char_kernel(x_anchor.unsqueeze(0), g_neg, delta)  # (N,B)

        # ------------------------------------------------------------
        # 6) Temperature‑free contrastive loss
        # ------------------------------------------------------------
        # sp = log Σ_pos e^{κ⁺},  sn = log Σ_neg e^{κ⁻}
        sp = torch.logsumexp(sim_pos, dim=0)
        sn = torch.logsumexp(sim_neg, dim=0)
        loss = -(sp - sn).mean()

        # ------------------------------------------------------------
        # 7) Diagnostics — for logs/telemetry
        # ------------------------------------------------------------
        pos_score = torch.exp(sim_pos).mean().item()
        neg_score = torch.exp(sim_neg).mean().item()

        return loss, pos_score, neg_score
