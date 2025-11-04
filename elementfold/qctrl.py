# ElementFold · qctrl.py
# Quantum‑control helpers that turn the coherence click δ⋆ into gate schedules.
# Read this like a recipe:
#   1) θ⋆(δ⋆) is the base rotation angle: one “click” on the unit circle → one gate quantum.
#   2) We compile integer multiples of θ⋆ and (optionally) wrap them into (−π, π] for hardware.
#   3) We build palindromic sequences (… A, B, …, B, A …) to cancel many systematic errors.
#   4) We offer an integer‑dithering compiler that approximates any target angle φ with a short
#      sequence of integers {m_i} whose average multiple ≈ φ/θ⋆ (no analog trims needed).

from __future__ import annotations                    # ↻ forward annotations on older Python
import math                                           # ✴ π, cos, modulo
from typing import Sequence, List, Dict, Any          # light hints for clarity

TAU = 2.0 * math.pi                                   # τ = 2π (nice for rotations)


# ——————————————————————————————————————————————————————————————————————————————
# 1) Base angle and safe angle arithmetic
# ——————————————————————————————————————————————————————————————————————————————

def theta_star(delta: float) -> float:                # θ⋆(δ⋆) = 2π·δ⋆
    """
    Convert a coherence click δ⋆ into the base gate angle θ⋆.
    This is the “quantum of rotation” used across schedules.
    """
    return TAU * float(delta)                         # θ⋆ in radians


def wrap_pi(theta: float) -> float:                   # ↻ wrap angle into (−π, π]
    """
    Reduce any angle θ into the principal interval (−π, π].
    Many control stacks assume phases live here to avoid ambiguity.
    """
    t = (float(theta) + math.pi) % TAU                # shift → modulo τ
    return t - math.pi                                # shift back into (−π, π]


# ——————————————————————————————————————————————————————————————————————————————
# 2) Compile integer multiples of θ⋆ (with optional 2π wrapping)
# ——————————————————————————————————————————————————————————————————————————————

def compile_multiples(m: Sequence[int | float], delta: float, mod: bool = True) -> List[float]:
    """
    Turn a list of integer (or real) multiples m into actual angles m·θ⋆.
    If mod=True, wrap into (−π, π] which many backends expect.
    """
    th = theta_star(delta)                            # θ⋆ in radians
    angles = [float(k) * th for k in m]               # raw angles (no wrapping)
    if mod:
        angles = [wrap_pi(a) for a in angles]         # hardware range
    return angles                                     # list[float] radians


# ——————————————————————————————————————————————————————————————————————————————
# 3) Palindromic schedules (echo‑style error cancellation)
# ——————————————————————————————————————————————————————————————————————————————

def palindrome(seq: Sequence[float], include_center: bool = False, sign: str = "same") -> List[float]:
    """
    Build a palindromic sequence:
        seq_pal = [s0, s1, …, sN,  (maybe sN again),  sN−1, …, s1, s0]
    Options:
      • include_center=False  → do not duplicate the center element (standard echo).
      • sign='same' | 'flip'  → mirror with same sign or with negated sign (spin‑echo style).

    Why: Palindromes cancel many first‑order systematics (e.g., static detuning) by symmetry.
    """
    core = list(seq)                                  # forward half
    if not core:                                      # edge‑case: empty
        return []

    # Choose the mirrored tail: drop the last element unless the caller asks to include it.
    tail_src = core if include_center else core[:-1]
    tail = list(reversed(tail_src))                   # reverse to make the mirror
    if sign == "flip":
        tail = [(-x) for x in tail]                   # sign‑flip for spin‑echo patterns
    elif sign != "same":
        raise ValueError("sign must be 'same' or 'flip'")

    return core + tail                                # palindromic schedule


# ——————————————————————————————————————————————————————————————————————————————
# 4) Integer dithering: approximate any φ with short integer multiples of θ⋆
# ——————————————————————————————————————————————————————————————————————————————

def _even_ones_distribution(ones: int, length: int) -> List[int]:
    """
    Evenly distribute `ones` over `length` slots using a Bresenham‑like scheme.
    Returns a 0/1 list of length `length` with exactly `ones` ones, spaced as uniformly as possible.
    """
    if ones <= 0:
        return [0] * length
    if ones >= length:
        return [1] * length
    out: List[int] = []
    prev = 0
    for i in range(1, length + 1):
        curr = round(i * ones / length)               # ideal cumulative count at position i
        bit = curr - prev                             # 0 or 1 to keep totals exact
        out.append(int(bit))
        prev = curr
    return out


def dither_multiples(target_angle: float, delta: float, length: int) -> List[int]:
    """
    Compile a target angle φ into a length‑L list of integer multiples {m_i} such that
      average(m_i) ≈ φ/θ⋆
    with the minimum possible deviation at this length. We only use integers → no analog trims.

    Algorithm:
      1) Let a = φ/θ⋆. Split into floor/ceil: a = n + f, where n=floor(a), f∈[0,1).
      2) Choose exactly H ≈ round(f * L) entries equal to (n+1); the remaining (L−H) entries equal to n.
      3) Distribute those H “+1” evenly across the L slots (Bresenham‑like), minimizing local bias.

    Returns:
      list[int] of length L with values in {n, n+1}.
    """
    if length <= 1:                                   # trivial case → single rounded integer
        th = theta_star(delta)
        return [int(round(float(target_angle) / th))]

    th = theta_star(delta)
    a = float(target_angle) / th                      # desired multiple in “clicks”
    n = math.floor(a)                                 # base integer
    f = a - n                                         # fractional part in [0,1)

    H = int(round(f * length))                        # number of (n+1) entries to place
    bits = _even_ones_distribution(H, length)         # spread them evenly across L positions
    return [n + b for b in bits]                      # integers {n or n+1} with the right average


def compile_phase_schedule(
    phi: float,                                       # target rotation angle φ (radians)
    delta: float,                                     # coherence click δ⋆
    length: int | None = None,                        # dither length; if None/≤1 → single integer
    echo: bool = True,                                # build a palindromic echo by default
    include_center: bool = False,                     # whether to duplicate the center in the palindrome
    sign: str = "flip",                               # mirror sign policy: 'same' or 'flip'
    mod: bool = True,                                 # wrap angles into (−π, π] for hardware
) -> Dict[str, Any]:
    """
    Full compiler for a target angle φ:

      • Convert φ into an integer‑multiple schedule using length‑L dithering (optional).
      • Turn multiples into angles m·θ⋆ and wrap to (−π, π] if requested.
      • Optionally build a palindromic echo to cancel many systematics.

    Returns:
      {
        'theta_star': θ⋆,
        'multiples':  [m_i],                 # integers
        'angles':     [α_i],                 # α_i = m_i·θ⋆ (wrapped if mod=True)
        'palindromic': bool,                 # whether we applied palindrome()
        'echo_sign':  str,                   # 'same' or 'flip'
      }
    """
    th = theta_star(delta)                               # θ⋆ for this δ⋆
    if length is None or length <= 1:
        m = [int(round(float(phi) / th))]                # single integer multiple
    else:
        m = dither_multiples(phi, delta, int(length))    # evenly dithered integers

    angles = compile_multiples(m, delta, mod=mod)        # turn into actual angles (wrapped if requested)

    if echo and len(angles) >= 1:                        # palindromic echo schedule
        angles = palindrome(angles, include_center=include_center, sign=sign)

    return {
        "theta_star": th,
        "multiples": m,
        "angles": angles,
        "palindromic": bool(echo),
        "echo_sign": sign if echo else "none",
    }


__all__ = [
    "theta_star",
    "wrap_pi",
    "compile_multiples",
    "palindrome",
    "dither_multiples",
    "compile_phase_schedule",
]
