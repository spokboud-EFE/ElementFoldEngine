# ElementFold · align.py
# ============================================================
# Alignment head — teaches representations to stay phase-aligned
# on the δ★ (delta-star) circle, without a temperature parameter.
#
# Conceptual picture:
#   • Each sample x lives on a circular ledger measured in δ★ units.
#   • "Positive" examples are tiny, legal micro-shifts within one click.
#   • "Negative" examples are half-click offsets (±δ★/2), where similarity
#     should be minimal — the opposite seat on the circle.
#
# The cosine kernel κ(x,y)=cos[2π(x−y)/δ★] gives the exact periodic similarity.
# From it we build a contrastive loss:
#      L = − E_b [ log Σ_pos e^{κ⁺} − log Σ_neg e^{κ⁻} ]
# No temperature or margin is needed: δ★ itself fixes the scale.
# ============================================================

from __future__ import annotations
import math
import torch
import torch.nn as nn

# κ(x,y) = cos(2π(x−y)/δ★); defined in ledger.py
from .ledger import char_kernel


class AlignHead(nn.Module):
    """
    Temperature-free circular contrastive head.

    Parameters
    ----------
    delta : float
        The fixed click size δ★ (sets the periodicity of the circle).

    Forward signature
    -----------------
        loss, pos_score, neg_score = align(
            x, capacities,
            num_pos=2, num_neg=8,
            noise=0.02, neg_jitter=0.20
        )

    Inputs
    ------
    x : Tensor (B,) or (B,T)
        Ledger positions. If (B,T), they are averaged across T to get one
        effective phase per batch element.
    capacities : scalar, (B,), or broadcastable
        Per-sample seat capacities (integer counts C). If the shape
        doesn’t match {1,B}, we fall back to max(C) across the input.
    num_pos : int
        Number of positive micro-shifts per sample (small, ≈2).
    num_neg : int
        Number of half-click negatives per sample.
    noise : float
        Tiny additive noise in ledger units to break symmetry.
    neg_jitter : float
        Standard deviation (as fraction of δ★) for half-click jitter.

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
        # Many encoders produce a sequence of ledger points per sample.
        # For alignment we only need the mean phase per sample.
        if x.dim() > 1:
            x = x.mean(dim=1)
        B = int(x.shape[0])
        dev, dtype = x.device, x.dtype

        # ------------------------------------------------------------
        # 2) Capacity handling
        # ------------------------------------------------------------
        # Each sample may have a different seat capacity C (how many
        # sub-positions fit in one δ★). Convert all to a clean (B,) tensor.
        caps = capacities if torch.is_tensor(capacities) else torch.as_tensor(capacities)
        caps = caps.to(device=dev)
        if caps.numel() == 1:
            caps_b = caps.view(1).expand(B)
        elif caps.numel() == B:
            caps_b = caps
        else:
            # Mismatch: fallback to conservative max capacity.
            caps_b = caps.max().view(1).expand(B)
        # Force integer and non-zero.
        caps_b = caps_b.clamp_min(1).to(torch.int64)

        # Seat step per sample: Δ_b = δ★ / C_b
        delta = self.delta
        delta_t = torch.as_tensor(delta, device=dev, dtype=dtype)
        step_b = delta_t / caps_b.to(dtype)  # (B,)

        # ------------------------------------------------------------
        # 3) POSITIVES — small micro-shifts within one click
        # ------------------------------------------------------------
        # Restrict offsets to roughly ±δ★/6 so they remain on the same seat.
        M_b = torch.clamp(caps_b // 6, min=1)  # (B,)
        if num_pos < 1:
            num_pos = 1

        # For each positive, pick a random integer step m∈[1,M_b] and sign s∈{−1,+1}.
        u = torch.rand(num_pos, B, device=dev)
        m = torch.floor(u * M_b.unsqueeze(0).to(dtype=u.dtype)).to(torch.int64) + 1
        s_pos = (torch.randint(0, 2, (num_pos, B), device=dev, dtype=torch.int64) * 2 - 1)

        # Construct positive targets:
        #   g_pos = x + s·m·Δ + ε   (noise breaks degeneracy)
        g_pos = (
            x.unsqueeze(0).to(dtype)
            + (s_pos * m).to(dtype) * step_b.unsqueeze(0)
            + noise * torch.randn(num_pos, B, device=dev, dtype=dtype)
        )

        # ------------------------------------------------------------
        # 4) NEGATIVES — around half-click (±δ★/2)
        # ------------------------------------------------------------
        # Half-click moves are the “furthest apart” points on the circle.
        # Add a bit of jitter so negatives cover a small neighborhood.
        if num_neg < 1:
            num_neg = 1
        s_neg = (torch.randint(0, 2, (num_neg, B), device=dev, dtype=torch.int64) * 2 - 1).to(dtype)
        half = 0.5 * delta_t
        neg_sigma = float(neg_jitter) * delta
        g_neg = (
            x.unsqueeze(0).to(dtype)
            + s_neg * half
            + neg_sigma * torch.randn(num_neg, B, device=dev, dtype=dtype)
        )

        # ------------------------------------------------------------
        # 5) Similarity on the δ★ circle
        # ------------------------------------------------------------
        # The circular cosine kernel computes exact periodic distance.
        # Shapes: x[None,:] → (1,B); g_pos → (P,B); g_neg → (N,B)
        sim_pos = char_kernel(x.unsqueeze(0), g_pos, delta)  # (P,B), values in [−1,1]
        sim_neg = char_kernel(x.unsqueeze(0), g_neg, delta)  # (N,B)

        # ------------------------------------------------------------
        # 6) Temperature-free contrastive loss
        # ------------------------------------------------------------
        # Instead of using a temperature τ, δ★ provides the natural scale.
        # We aggregate positives and negatives separately via log-sum-exp:
        #     sp = log Σ_pos e^{κ⁺},  sn = log Σ_neg e^{κ⁻}
        sp = torch.logsumexp(sim_pos, dim=0)
        sn = torch.logsumexp(sim_neg, dim=0)
        loss = -(sp - sn).mean()

        # ------------------------------------------------------------
        # 7) Diagnostics — for logging or telemetry
        # ------------------------------------------------------------
        pos_score = torch.exp(sim_pos).mean().item()
        neg_score = torch.exp(sim_neg).mean().item()

        # Return full tuple for direct loss + monitoring.
        return loss, pos_score, neg_score
