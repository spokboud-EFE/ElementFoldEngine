# ElementFold · ledger.py
# This file is the “geometry of coherence.” It turns any real value x into a point on a circle of size δ⋆,
# gives you the nearest click (rung) and the leftover offset (residual), measures distances on that circle,
# and provides kernels and simple identities that every layer of the engine uses.

import torch, math  # We import PyTorch for tensors and math for π and basic constants.

# ————————————————————————————————————————————————
# Core circular mapping
# ————————————————————————————————————————————————

def phase(x, delta):
    """
    Map a real position x onto the unit circle using the click δ⋆ as the period.
    Returns a complex tensor on the unit circle, i.e., e^{i·2πx/δ⋆}.
    """
    a = (2 * math.pi / delta) * x            # Convert “distance along the ledger” into an angle in radians.
    one = torch.ones_like(a, dtype=a.dtype)  # Build a tensor of ones so we can form a complex number with magnitude 1.
    return torch.polar(one, a)               # Construct complex numbers from (radius=1, angle=a). This lives exactly on the circle.


def rung_residual(x, delta):
    """
    Split x into (k, r) where k is the integer number of clicks and r is the signed
    residual inside the current click. The residual is always in (−½δ⋆, ½δ⋆].
    """
    k = torch.floor((x / delta) + 0.5).to(torch.int64)  # Round x/δ⋆ to the nearest integer by adding 0.5 and taking floor.
    r = x - k.to(x.dtype) * delta                       # Remove k clicks from x to get the leftover within the current click.
    return k, r                                         # Return both pieces: the “lap count” k and the “within‑lap” offset r.


def half_click_margin(x, delta):
    """
    How far x is from the nearest boundary between clicks (½δ⋆ away from the center).
    Positive margin means “safe” (will round to the same k under small noise).
    """
    _, r = rung_residual(x, delta)                      # Get the within‑click residual.
    return (delta / 2) - r.abs()                        # Margin = half a click minus how far we currently are from center.


def snap_to_rung(x, delta):
    """
    Snap x onto its nearest click center k·δ⋆. Returns the snapped value, k, and r.
    """
    k, r = rung_residual(x, delta)                      # Use the same decomposition into integer clicks and residual.
    x_snap = k.to(x.dtype) * delta                      # Reconstruct the snapped position on the exact click center.
    return x_snap, k, r                                 # Provide all three: snapped x, integer rung, and residual.


# ————————————————————————————————————————————————
# Seats inside a click
# ————————————————————————————————————————————————

def seat_index(x, delta, C):
    """
    Continuous seat coordinate inside a click with capacity C (e.g., C=6 means hexagonal),
    measured in the range [0, C). This is the fractional seat position, not rounded to an integer.
    """
    return ((C / delta) * x).remainder(C)               # Scale x by C/δ⋆ and wrap into one revolution with remainder C.


def seat_index_int(x, delta, C):
    """
    Integer seat index (0,1,...,C−1) by rounding to the nearest seat within the click.
    Useful when you need a discrete label instead of a continuous coordinate.
    """
    s = (C / delta) * x                                 # Convert x to “seat units” so one click spans exactly C seats.
    idx = torch.floor(s + 0.5).to(torch.int64)          # Round to the nearest seat with the same trick as rung rounding.
    return torch.remainder(idx, int(C))                 # Wrap it into {0,...,C−1} so seat indices loop around cleanly.


# ————————————————————————————————————————————————
# Distances and averages on the circle
# ————————————————————————————————————————————————

def wrapped_distance(x, y, delta):
    """
    The shortest arc length between x and y on the circle of size δ⋆.
    Always returns a value in [0, ½δ⋆].
    """
    d = (x - y).remainder(delta)                        # First wrap the raw difference into [0, δ⋆).
    half = delta / 2                                    # Half a click is the boundary between “go left” and “go right”.
    d = torch.where(d > half, d - delta, d)             # If we overshot to the right, step one full circle left.
    d = torch.where(d < -half, d + delta, d)            # If we overshot to the left, step one full circle right.
    return d.abs()                                      # We only care about the arc length, not the direction.


def periodic_mean(x, delta):
    """
    The average of many points on a circle is not a simple arithmetic mean; it’s a circular mean.
    We average unit phases and convert the mean phase back to a position in [0, δ⋆).
    """
    ph = phase(x, delta)                                # Map each x to its unit complex point on the circle.
    ph_mean = ph.mean()                                 # Average in the complex plane to keep angles consistent.
    ang = torch.angle(ph_mean)                          # Extract the angle of the average complex number in radians.
    pos = (ang * delta) / (2 * math.pi)                 # Convert the angle back to “ledger distance” in units of δ⋆.
    return pos.remainder(delta)                         # Wrap into [0, δ⋆) to stay on the ledger.


def periodic_lerp(x, y, w, delta):
    """
    Interpolate from x to y along the shortest arc on the circle. The weight w∈[0,1]
    chooses the point w of the way along that shortest path.
    """
    d = (y - x)                                         # Start with the direct difference.
    # Move the difference into the shortest direction by wrapping into (−½δ⋆, ½δ⋆].
    d = (d + delta / 2).remainder(delta) - (delta / 2)  # This centers the residual so it is symmetric around zero.
    return (x + w * d).remainder(delta)                 # Take a partial step and wrap to stay on the circle.


# ————————————————————————————————————————————————
# Circular kernels (how “in tune” two positions are)
# ————————————————————————————————————————————————

def char_kernel(x, y, delta):
    """
    The simplest exact circular similarity: cos(2π(x−y)/δ⋆).
    Equals 1 when x and y coincide on the circle, and −1 when they are half a click apart.
    """
    return torch.cos(2 * math.pi * (x - y) / delta)     # Compute cosine of the normalized angular difference.


def vm_kernel(x, y, delta):
    """
    A smooth, positive kernel on the circle: exp(cos(2π(x−y)/δ⋆)).
    Often called a von Mises kernel; it respects periodicity exactly.
    """
    return torch.exp(torch.cos(2 * math.pi * (x - y) / delta))  # Turn cosine into a soft, always‑positive similarity.


# ————————————————————————————————————————————————
# Invariants and simple checks
# ————————————————————————————————————————————————

def invariants(U):
    """
    Given a potential U, the four canonical channels are exponentials of ±U and ±2U.
    We return three of them: Γ (clock), n (index), I (intensity).
    """
    Gam = torch.exp(+U)                                 # Clock factor grows like e^{+U}.
    nidx = torch.exp(-2 * U)                            # Refractive index grows like e^{−2U}.
    I = torch.exp(+2 * U)                               # Intensity grows like e^{+2U}.
    return Gam, nidx, I                                 # These satisfy simple multiplicative identities.


def check_identities(U, eps=1e-6):
    """
    Verify the gauge‑free identities:
      Γ · n^{1/2} = 1  and  I · n = 1.
    We return the absolute deviations from 1 so you can see how well the data obeys the law.
    """
    Gam, n, I = invariants(U)                           # Compute the channels from U.
    lhs1 = Gam * torch.sqrt(n)                          # Left side of the first identity.
    lhs2 = I * n                                        # Left side of the second identity.
    err1 = (lhs1 - 1.0).abs().max().item()             # Max absolute deviation for the first identity.
    err2 = (lhs2 - 1.0).abs().max().item()             # Max absolute deviation for the second identity.
    ok1 = err1 <= eps                                   # Within tolerance?
    ok2 = err2 <= eps                                   # Within tolerance?
    return {"err_Gam_nhalf": err1, "err_I_n": err2, "ok": bool(ok1 and ok2)}  # Report both errors and a combined pass flag.


def kappa(x, delta):
    """
    Phase concentration κ in [0,1]: how well a set of points aligns on the circle.
    κ≈1 means all phases point the same way; κ≈0 means they are spread out.
    """
    ph = phase(x, delta)                                # Turn positions into unit complex numbers.
    return torch.abs(ph.mean()).item()                  # The magnitude of the mean complex vector is the concentration.


def p_half(x, delta, eps=1e-9):
    """
    Fraction of samples that lie on or beyond the half‑click boundary (the seat boundary).
    Larger values mean you are skimming the boundary and likely to “flip” to the next click.
    """
    _, r = rung_residual(x, delta)                      # Get residuals for each sample.
    return (r.abs() >= (delta / 2 - eps)).float().mean().item()  # Count boundary touches and average to a fraction.
