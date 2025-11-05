# ElementFold · align.py
# ──────────────────────────────────────────────────────────────────────────────
# Goal (plain words)
# ------------------
# Teach the model to keep phases aligned on the δ⋆ circle **without** a fiddly
# temperature. We compare each sample x against:
#   • positives: tiny, legal seat steps that stay within one click (micro‑shifts),
#   • negatives: half‑click moves (±δ⋆/2), where similarity should be minimal.
#
# Geometry (exact, not approximate)
# ---------------------------------
# The cosine character kernel is the exact circular similarity:
#     κ(x,y) = cos(2π(x−y)/δ⋆).
# Turning κ into exp(κ) gives a soft score that lives in [e^{−1}, e^{+1}].
#
# Loss (temperature‑free)
# -----------------------
# For each sample we aggregate positives and negatives via log‑sum‑exp and take
#    ℒ = − E_b [ log Σ_pos e^{κ⁺} − log Σ_neg e^{κ⁻} ].
# δ⋆ sets the natural scale; no temperature knob is needed.

from __future__ import annotations
import math
import torch
import torch.nn as nn

from .ledger import char_kernel  # κ(x,y) = cos(2π(x−y)/δ⋆)


class AlignHead(nn.Module):
    """
    A tiny, temperature‑free contrastive head on the δ⋆ circle.

    Construction:
        AlignHead(delta)

    Call:
        loss, pos_score, neg_score = align(x, capacities, num_pos=2, num_neg=8, noise=0.02)

    Where:
        • x:           (B,) or (B,T) ledger positions; if (B,T) we reduce to (B,) by mean over T.
        • capacities:  per‑sample seat capacities (C); can be a scalar, (B,), or “other”.
                       If shape does not match {1, B}, we fall back to max(C) across the input.
        • num_pos:     how many positive micro‑shifts to sample per sample (small is fine).
        • num_neg:     how many negatives near ±δ⋆/2 per sample.
        • noise:       small additive jitter in *ledger units* (stays tiny).
        • neg_jitter:  std of the half‑click jitter as a fraction of δ⋆ (moderate, default 0.20).

    Returns:
        (loss, pos_score, neg_score)
            loss:       scalar tensor
            pos_score:  float, average exp(κ) over positive pairs (higher is better)
            neg_score:  float, average exp(κ) over negative pairs (lower is better)
    """

    def __init__(self, delta: float):
        super().__init__()
        self.delta = float(delta)

    def forward(
        self,
        x: torch.Tensor,                   # (B,) or (B,T)
        capacities,                        # scalar, (B,), or broadcastable; fallback = max(C)
        num_pos: int = 2,
        num_neg: int = 8,
        noise: float = 0.02,               # tiny additive noise in ledger units
        neg_jitter: float = 0.20,          # fraction of δ⋆ for half‑click jitter (keeps negatives “near” ±½δ⋆)
    ):
        # ——— 1) Reduce sequences to one phase per sample (if needed) ————————
        if x.dim() > 1:
            x = x.mean(dim=1)              # (B,T) → (B,)
        B = int(x.shape[0])
        dev = x.device
        dtype = x.dtype

        # ——— 2) Capacity handling (robust, user‑friendly) ————————————————
        # Make 'caps_b' a (B,) vector of per‑sample capacities.
        caps = capacities if torch.is_tensor(capacities) else torch.as_tensor(capacities)
        caps = caps.to(device=dev)
        if caps.numel() == 1:
            caps_b = caps.view(1).expand(B)
        elif caps.numel() == B:
            caps_b = caps
        else:
            # Mismatch: use a conservative upper bound (max capacity).
            caps_b = caps.max().view(1).expand(B)
        caps_b = caps_b.clamp_min(1).to(torch.int64)     # C ≥ 1, keep it integral

        # Seat step per sample: Δ_b = δ⋆ / C_b  (shape (B,))
        delta = self.delta
        delta_t = torch.as_tensor(delta, device=dev, dtype=dtype)
        step_b = (delta_t / caps_b.to(dtype))            # Δ per sample

        # ——— 3) POSITIVES: micro‑shifts (± small seat counts) within a click ——
        # We restrict to tiny offsets so positives really are “close” on the circle.
        # Let M_b = max(1, ⌊C_b/6⌋): at most ~δ⋆/6 away.
        M_b = torch.clamp(caps_b // 6, min=1)           # (B,)
        if num_pos < 1:
            num_pos = 1

        # Sample magnitudes m ∈ {1,…,M_b} and random signs s ∈ {−1,+1} per (pos, sample).
        # Use a uniform draw via continuous trick: floor(u*M_b)+1, with u∈[0,1).
        u = torch.rand(num_pos, B, device=dev)
        m = torch.floor(u * M_b.unsqueeze(0).to(dtype=u.dtype)).to(torch.int64) + 1
        s_pos = (torch.randint(0, 2, (num_pos, B), device=dev, dtype=torch.int64) * 2 - 1)

        # Build positive targets g_pos = x + s·m·Δ + ε, broadcasting Δ per sample.
        # Noise is tiny, unscaled (ledger units); shape aligns with (num_pos, B).
        g_pos = (
            x.unsqueeze(0).to(dtype)
            + (s_pos * m).to(dtype) * step_b.unsqueeze(0)
            + noise * torch.randn(num_pos, B, device=dev, dtype=dtype)
        )

        # ——— 4) NEGATIVES: near half‑click (±½δ⋆) with moderate jitter ————————
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

        # ——— 5) Similarities on the circle (exact periodic geometry) ————————
        # Shapes: x[None,:] → (1,B); g_pos → (P,B); g_neg → (N,B)
        sim_pos = char_kernel(x.unsqueeze(0), g_pos, delta)  # (P,B) in [−1,1]
        sim_neg = char_kernel(x.unsqueeze(0), g_neg, delta)  # (N,B) in [−1,1]

        # ——— 6) Temperature‑free contrast using log‑sum‑exp (stable) ————————
        # sp = log Σ_pos exp(sim_pos), sn = log Σ_neg exp(sim_neg)
        sp = torch.logsumexp(sim_pos, dim=0)                 # (B,)
        sn = torch.logsumexp(sim_neg, dim=0)                 # (B,)
        loss = -(sp - sn).mean()

        # Diagnostic scores (nice to log): average exp(κ⁺) and exp(κ⁻).
        pos_score = torch.exp(sim_pos).mean().item()
        neg_score = torch.exp(sim_neg).mean().item()

        return loss, pos_score, neg_score
