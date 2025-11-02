# ElementFold · telemetry.py
# Telemetry tells us if the system is in tune. We take raw ledger values X and report
# a few small, meaningful numbers that summarize coherence on the δ⋆ circle:
#   • κ (kappa): how tightly phases align (1 = perfectly aligned, 0 = spread out)
#   • p_half: how often samples touch the half‑click boundary (risk of flipping rungs)
#   • margins: safety gap to the boundary (mean / min)
#   • residual stats: how wide the residuals are around their centers
# If X is a sequence per sample (shape (B,T)), we also report how big the steps are along T.

import torch, math                           # PyTorch for tensors; math for π (used in angle→position conversion)
from .ledger import phase, rung_residual, wrapped_distance  # Circular tools from the ledger

def measure(X, delta, eps=1e-9, detail=False):
    """
    Compute coherence diagnostics on the δ⋆ circle.

    Args:
        X:     Tensor of ledger values. Shape can be:
               • (B,)  — one value per sample, or
               • (B,T) — a short sequence per sample; we summarize each row.
        delta: The fundamental click δ⋆ that defines the circle size.
        eps:   A tiny tolerance when checking half‑click boundaries.
        detail:If True and X has shape (B,T), also report step statistics along T.

    Returns:
        dict with scalar floats (and a few sizes if detail=True):
            {
              'kappa':      phase concentration in [0,1],
              'p_half':     fraction at or beyond half‑click boundary,
              'margin_mean':average safety gap to the boundary (larger is safer),
              'margin_min': smallest safety gap (how close the worst sample gets),
              'resid_std':  standard deviation of residuals r,
              'phase_mean': average phase location on [0, δ⋆),
              # if detail and X is (B,T):
              'step_mean':  average wrapped step size between neighboring seats,
              'step_std':   std of those steps,
              'B': int(B), 'T': int(T)
            }
    """
    # ——— 1) Reduce sequences (if any) to a single ledger value per sample ————————
    if X.dim() == 2:                         # If we have (B,T) values per sample …
        x = X.mean(dim=1)                    # … average over T to get a representative phase per sample.
    else:                                    # Otherwise X is already (B,) or (T,)
        x = X                                # … use it as‑is (broadcasts fine if (T,)).

    # ——— 2) Phase concentration κ on the unit circle ————————————————
    ph = phase(x, delta)                     # Map each x to a unit complex: e^{i·2πx/δ⋆}.
    ph_mean = ph.mean()                      # Average those complex numbers to find the centroid on the circle.
    kappa = torch.abs(ph_mean).item()        # κ = |centroid|; near 1 means phases are aligned.

    # We also report where that mean phase sits on [0, δ⋆) (useful to detect global drift).
    ang = torch.angle(ph_mean)               # Angle of the mean complex number in radians (−π, π].
    phase_mean = float(((ang * delta) / (2 * math.pi)) % delta)  # Convert angle back into δ⋆ units and wrap.

    # ——— 3) Residuals within a click and boundary contact rate —————————
    k, r = rung_residual(x, delta)           # x = k·δ⋆ + r, where r ∈ (−½δ⋆, ½δ⋆].
    # p_half counts how often we are at or past the half‑click boundary (danger zone for rung flips).
    p_half = (r.abs() >= (delta / 2 - eps)).float().mean().item()

    # Safety margin to the boundary: m = ½δ⋆ − |r| (clamped at 0 so negatives don’t average away the warning).
    margin = (delta / 2) - r.abs()           # Positive = safe space left; negative would mean “already over the line”.
    margin_clamped = torch.clamp_min(margin, 0.0)  # Never let negative margins hide risk in the average.
    margin_mean = margin_clamped.mean().item()     # Average safety gap across samples.
    margin_min = margin.min().item()               # Smallest (possibly negative) margin — the worst case.

    # Residual spread: how wide r values are around their centers (0 means everyone sits exactly on centers).
    resid_std = (r.std(unbiased=False).item() if r.numel() > 1 else 0.0)

    # Prepare the basic report.
    report = {
        "kappa": kappa,                     # Phase alignment strength.
        "p_half": p_half,                   # Boundary touch frequency.
        "margin_mean": margin_mean,         # Average safety gap.
        "margin_min": margin_min,           # Worst safety gap.
        "resid_std": resid_std,             # Spread of residuals.
        "phase_mean": phase_mean,           # Average phase position (for drift checks).
    }

    # ——— 4) Optional step statistics along T when sequences are provided ———————
    if detail and X.dim() == 2 and X.size(1) > 1:  # Only meaningful if we had sequences (B,T) with T>1
        # Wrapped steps between neighbors measure how “bumpy” the path is along the sequence dimension.
        d = wrapped_distance(X[:, 1:], X[:, :-1], delta)  # Shortest circular distance seat‑to‑seat.
        step_mean = d.mean().item()                       # Average step magnitude across all rows and positions.
        step_std = d.std(unbiased=False).item()           # How variable those steps are.
        report.update({
            "step_mean": step_mean,                       # Typical local movement on the circle.
            "step_std": step_std,                         # Variability of that movement.
            "B": int(X.size(0)),                          # Number of sequences examined.
            "T": int(X.size(1)),                          # Sequence length per sample (as provided).
        })

    return report                                        # One small dict to judge coherence at a glance.
