# ElementFold · align.py
# Goal: teach the model to keep phases aligned on the δ⋆ circle without a tunable temperature.
# We do this by comparing each sample x to:
#   • positives: legal micro‑shifts inside the same click (seat steps), plus tiny noise,
#   • negatives: half‑click moves (±δ⋆/2), where similarity should be minimal.
# The cosine character kernel acts as the exact circular similarity; exp() turns it into a soft score.

import torch, torch.nn as nn, math           # Tensors, modules, and π for clarity (π used via ledger if needed)
from .ledger import char_kernel              # κ_char(x,y) = cos(2π(x−y)/δ⋆) — exact periodic similarity


class AlignHead(nn.Module):                  # A small head that produces a temperature‑free contrastive loss
    def __init__(self, delta):               # Accept δ⋆ so geometry is fixed by physics, not by tuning
        super().__init__()                   # Standard module init
        self.delta = float(delta)            # Cache δ⋆ as float for speed and clarity

    def forward(
        self,
        x,                                   # Input ledger positions; shape (B,) or (B,T) before reduction
        capacities,                          # Seat capacities; scalar, (B,), or any broadcastable form
        num_pos=2,                           # How many positive micro‑shifts per sample
        num_neg=8,                           # How many negative half‑click shifts per sample
        noise=0.02,                          # Small additive jitter to avoid degenerate symmetry ties
    ):
        # ——— 1) Make x a simple (B,) vector of ledger positions ————————————————
        if x.dim() > 1:                      # If we are given a sequence (B,T) of ledger scalars …
            x = x.mean(dim=1)                # … average over time to get a single representative phase per sample.
        B = x.shape[0]                       # Batch size for all sampling below.

        # ——— 2) Build a per‑sample capacity vector caps_b of length B ————————————
        caps = capacities if torch.is_tensor(capacities) else torch.as_tensor(capacities, device=x.device)
        caps = caps.to(device=x.device)      # Move to the same device as x for arithmetic
        if caps.numel() == 1:                # If one number (e.g., C=6) was given …
            caps_b = caps.view(1).expand(B)  # … broadcast it to every sample in the batch.
        elif caps.numel() == B:              # If a capacity per sample was provided …
            caps_b = caps                    # … use it as‑is.
        else:                                # Otherwise (mismatch), fall back to a safe upper bound …
            caps_b = caps.max().view(1).expand(B)  # … by using the maximum capacity for all samples.
        caps_b = caps_b.clamp_min(1)         # Never allow zero capacity; keep geometry well‑defined.

        # Compute the per‑sample seat step Δ_b = δ⋆ / C_b as a (B,) vector.
        invC = (self.delta / caps_b.float()) # This is how far one seat moves inside the click for each sample.

        # ——— 3) Sample POSITIVES: legal micro‑shifts within the same click —————————
        Cmax = int(caps_b.max().item())      # We sample in a unified range and then wrap per sample.
        if Cmax < 1:                         # Guard: if something went wrong with capacities, default to 1.
            Cmax = 1
        # Draw integer seat offsets a ∈ {0..Cmax−1} for each positive and sample; shape (num_pos, B).
        a = torch.randint(0, Cmax, (num_pos, B), device=x.device)
        a = torch.remainder(a, caps_b.unsqueeze(0))            # Wrap per sample to its own capacity
        a = a.to(x.dtype)                                      # Use same dtype as x for arithmetic below
        # Add those steps to x with tiny noise ε ~ N(0, noise²). Shapes broadcast to (num_pos, B).
        g_pos = x.unsqueeze(0) + a * invC.unsqueeze(0) + noise * torch.randn_like(a)

        # ——— 4) Sample NEGATIVES: half‑click away (±δ⋆/2), plus moderate noise ——————
        half = 0.5 * self.delta                                # The place of minimal cosine similarity on the circle
        # Draw random signs s ∈ {−1,+1} for each negative and sample; shape (num_neg, B).
        s = (torch.randint(0, 2, (num_neg, B), device=x.device) * 2 - 1).to(x.dtype)
        # Add ±½δ⋆ plus a broader jitter to avoid trivial alignment at exact antipodes.
        g_neg = x.unsqueeze(0) + s * half + (0.25 * self.delta) * torch.randn_like(s)

        # ——— 5) Similarities via the character kernel (exact circular geometry) —————
        # Broadcasting: x[None,:] is (1,B); g_pos is (num_pos,B); result is (num_pos,B)
        sim_pos = char_kernel(x.unsqueeze(0), g_pos, self.delta)  # High when aligned modulo the click
        sim_neg = char_kernel(x.unsqueeze(0), g_neg, self.delta)  # Low (≈−1) near ±½δ⋆
        # Turn similarities into soft scores in [e^{−1}, e^{+1}] and reduce over the positives/negatives.
        sp = sim_pos.exp().sum(dim=0)                            # Σ_pos exp(κ⁺) → (B,)
        sn = sim_neg.exp().sum(dim=0)                            # Σ_neg exp(κ⁻) → (B,)

        # ——— 6) Temperature‑free contrastive objective ————————————————
        # Instead of dividing by a temperature τ, we rely on δ⋆ as the physical scale baked into κ_char.
        num = sp.clamp_min(1e-9)                                  # Safety: avoid log(0)
        den = sn.clamp_min(1e-9)                                  # Safety: avoid log(0)
        loss = -(num / den).log().mean()                          # ℒ = −E_b[log(Σ⁺/Σ⁻)]

        # ——— 7) Return both scalar loss and simple diagnostics ———————————————
        return loss, sp.mean().item(), sn.mean().item()           # Loss, average positive score, average negative score
