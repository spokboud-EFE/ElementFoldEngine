# ElementFold · qctrl.py
# Quantum‑control helpers that turn the coherence click δ⋆ into gate schedules.
# Read this like a recipe:
#   1) θ⋆(δ⋆) is the base rotation angle: one “click” on the unit circle → one gate quantum.
#   2) We compile integer multiples of θ⋆ and (optionally) wrap them into (−π, π] for hardware.
#   3) We build palindromic sequences (… A, B, …, B, A …) to cancel many systematic errors.
#   4) We offer an integer‑dithering compiler that approximates any target angle φ with a short
#      sequence of integers {m_i} whose average multiple ≈ φ/θ⋆ (no analog trims needed).
#
# Extras in this version (still dependency‑free):
#   • schedule_sum / schedule_stats — quick diagnostics (net rotation; error vs. target).
#   • Stronger input guards and clear edge‑case semantics (empty/length=1 paths).
#
# Notes on “echo” palindromes:
#   • sign='same'   → mirror pulses with the same sign. Net rotation roughly doubles.
#   • sign='flip'   → mirror pulses with negated sign (spin‑echo style). With include_center=False,
#                     the net rotation equals the *last* element of the core sequence. This pattern
#                     cancels many quasi‑static errors while preserving one “effective” pulse.

from __future__ import annotations
import math
from typing import Sequence, List, Dict, Any, Tuple

TAU = 2.0 * math.pi  # τ = 2π


# ──────────────────────────────────────────────────────────────────────────────
# 1) Base angle and safe angle arithmetic
# ──────────────────────────────────────────────────────────────────────────────

def theta_star(delta: float) -> float:
    """
    θ⋆(δ⋆) = 2π·δ⋆.
    Convert a coherence click δ⋆ into the base gate angle θ⋆ (radians).
    """
    return TAU * float(delta)


def wrap_pi(theta: float) -> float:
    """
    Wrap any angle θ into the principal interval (−π, π].
    """
    t = (float(theta) + math.pi) % TAU
    return t - math.pi


# ──────────────────────────────────────────────────────────────────────────────
# 2) Compile integer (or real) multiples of θ⋆ (with optional 2π wrapping)
# ──────────────────────────────────────────────────────────────────────────────

def compile_multiples(m: Sequence[int | float], delta: float, mod: bool = True) -> List[float]:
    """
    Turn a list of multiples m into actual angles m·θ⋆.
    If mod=True, each angle is wrapped into (−π, π] (many backends expect this).
    """
    th = theta_star(delta)
    angles = [float(k) * th for k in m]
    if mod:
        angles = [wrap_pi(a) for a in angles]
    return angles


# ──────────────────────────────────────────────────────────────────────────────
# 3) Palindromic schedules (echo‑style error cancellation)
# ──────────────────────────────────────────────────────────────────────────────

def palindrome(seq: Sequence[float], include_center: bool = False, sign: str = "same") -> List[float]:
    """
    Build a palindromic sequence:
        seq_pal = [s0, s1, …, sN,  (maybe sN again),  sN−1, …, s1, s0]

    Args:
      include_center:  If False (default), we do not duplicate the center element.
      sign:            'same'  → mirrored tail keeps the same sign,
                       'flip'  → mirrored tail is negated (spin‑echo style).

    Returns:
      List[float] palindromic schedule.

    Important: With sign='flip' and include_center=False, the net sum equals the *last*
    element of the forward half (∑core − ∑core[:-1] = core[-1]). This is often desirable
    to preserve one effective rotation while cancelling many systematics symmetrically.
    """
    core = list(seq)
    if not core:
        return []
    tail_src = core if include_center else core[:-1]
    tail = list(reversed(tail_src))
    if sign == "flip":
        tail = [(-x) for x in tail]
    elif sign != "same":
        raise ValueError("sign must be 'same' or 'flip'")
    return core + tail


# ──────────────────────────────────────────────────────────────────────────────
# 4) Integer dithering: approximate any φ with short integer multiples of θ⋆
# ──────────────────────────────────────────────────────────────────────────────

def _even_ones_distribution(ones: int, length: int) -> List[int]:
    """
    Evenly distribute `ones` over `length` slots using a Bresenham‑like scheme.
    Returns a 0/1 list of length `length` with exactly `ones` ones, spaced uniformly.
    """
    L = max(0, int(length))
    H = max(0, min(int(ones), L))
    if L == 0:
        return []
    if H == 0:
        return [0] * L
    if H == L:
        return [1] * L
    out: List[int] = []
    prev = 0
    for i in range(1, L + 1):
        curr = round(i * H / L)   # ideal cumulative count at position i
        out.append(int(curr - prev))
        prev = curr
    return out


def dither_multiples(target_angle: float, delta: float, length: int) -> List[int]:
    """
    Compile a target angle φ into a length‑L list of integer multiples {m_i} such that
      average(m_i) ≈ φ/θ⋆
    with the minimum possible deviation at this length. We only use integers → no analog trims.

    Algorithm (Bresenham‑style):
      1) Let a = φ/θ⋆. Split into floor/ceil: a = n + f, where n=floor(a), f∈[0,1).
      2) Choose exactly H ≈ round(f * L) entries equal to (n+1); the remaining (L−H) entries equal to n.
      3) Distribute those H “+1” evenly across the L slots to minimize local bias.

    Edge cases:
      • length<=1 → returns a single rounded multiple [round(φ/θ⋆)].

    Returns:
      list[int] of length L with values in {n, n+1}.
    """
    L = int(length)
    if L <= 1:
        th = theta_star(delta)
        return [int(round(float(target_angle) / th))]
    th = theta_star(delta)
    a = float(target_angle) / th
    n = math.floor(a)
    f = a - n
    H = int(round(f * L))                 # how many (n+1) entries
    bits = _even_ones_distribution(H, L)  # spread them uniformly
    return [n + b for b in bits]


# ──────────────────────────────────────────────────────────────────────────────
# 5) Compiler + diagnostics
# ──────────────────────────────────────────────────────────────────────────────

def schedule_sum(angles: Sequence[float], mod: bool = False) -> float:
    """
    Sum a sequence of angles. If mod=True, wrap the sum into (−π, π].
    """
    total = float(sum(float(a) for a in angles))
    return wrap_pi(total) if mod else total


def schedule_stats(angles: Sequence[float], phi_target: float | None = None) -> Dict[str, float]:
    """
    Small diagnostic: report total (unwrapped) and wrapped sums, plus error vs. a target φ if provided.
    """
    total = schedule_sum(angles, mod=False)
    total_wrapped = schedule_sum(angles, mod=True)
    out: Dict[str, float] = {
        "sum_rad": total,
        "sum_wrapped_rad": total_wrapped,
        "sum_deg": math.degrees(total),
        "sum_wrapped_deg": math.degrees(total_wrapped),
    }
    if phi_target is not None:
        err = total - float(phi_target)
        err_wrapped = wrap_pi(total - float(phi_target))
        out.update({
            "err_rad": err,
            "err_deg": math.degrees(err),
            "err_wrapped_rad": err_wrapped,
            "err_wrapped_deg": math.degrees(err_wrapped),
        })
    return out


def compile_phase_schedule(
    phi: float,                      # target rotation angle φ (radians)
    delta: float,                    # coherence click δ⋆
    length: int | None = None,       # dither length; if None/≤1 → single integer
    echo: bool = True,               # build a palindromic echo by default
    include_center: bool = False,    # whether to duplicate the center in the palindrome
    sign: str = "flip",              # mirror sign policy: 'same' or 'flip'
    mod: bool = True,                # wrap pulse angles into (−π, π] for hardware
) -> Dict[str, Any]:
    """
    Full compiler for a target angle φ:

      • Convert φ into an integer‑multiple schedule using length‑L dithering (optional).
      • Turn multiples into angles m·θ⋆ and wrap to (−π, π] if requested.
      • Optionally build a palindromic echo to cancel many systematics.
      • Return light diagnostics, including net rotation.

    Important echo note:
      With sign='flip' and include_center=False, the palindrome’s *net* rotation equals the
      last element of the forward half. If you want the *sum* to approximate φ directly,
      set echo=False (or use sign='same' to roughly double the core sum).

    Returns:
      {
        'theta_star':     θ⋆,
        'multiples':      [m_i],                 # integers from dither (or single rounded integer)
        'angles':         [α_i],                 # α_i = m_i·θ⋆ (wrapped if mod=True; palindromic if echo=True)
        'palindromic':    bool,                  # whether palindrome() was applied
        'echo_sign':      'same'|'flip'|'none',
        'sum_rad':        float,                 # net sum of angles (unwrapped)
        'sum_wrapped_rad':float,                 # net sum wrapped into (−π, π]
        'err_rad':        float,                 # (optional) error vs φ (unwrapped)
        'err_wrapped_rad':float,                 # (optional) wrapped error vs φ
      }
    """
    th = theta_star(delta)

    # 1) Integer multiples
    if length is None or length <= 1:
        m = [int(round(float(phi) / th))]
    else:
        m = dither_multiples(phi, delta, int(length))

    # 2) Angles for the “core” subsequence
    core_angles = compile_multiples(m, delta, mod=mod)

    # 3) Optional palindrome (echo packaging)
    angles = core_angles
    if echo and len(core_angles) >= 1:
        angles = palindrome(core_angles, include_center=include_center, sign=sign)

    # 4) Diagnostics
    stats = schedule_stats(angles, phi_target=float(phi))

    out: Dict[str, Any] = {
        "theta_star": th,
        "multiples": m,
        "angles": angles,
        "palindromic": bool(echo),
        "echo_sign": sign if echo else "none",
        "sum_rad": stats["sum_rad"],
        "sum_wrapped_rad": stats["sum_wrapped_rad"],
        "err_rad": stats.get("err_rad", 0.0),
        "err_wrapped_rad": stats.get("err_wrapped_rad", wrap_pi(stats["sum_rad"] - float(phi))),
    }
    return out


__all__ = [
    "theta_star",
    "wrap_pi",
    "compile_multiples",
    "palindrome",
    "dither_multiples",
    "schedule_sum",
    "schedule_stats",
    "compile_phase_schedule",
]
