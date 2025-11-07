# ElementFold · variational.py
# ============================================================
# Variational Ledger — convex energy that keeps the ledger evenly spaced.
#
# Physical picture
# ----------------
# • Each block (a click of width δ★) has C_b seats.
# • Inside a block, seats are spaced by Δ_b = δ★ / C_b.
# • Across blocks, the first seat advances by exactly δ★:
#       X_{b+1,0} − X_{b,0} = δ★
# • The last seat wraps to the first seat with target (Δ_b − δ★),
#   not Δ_b — this preserves global phase coherence.
#
# Gauge:
#   Global translations X → X + const are unphysical.  With gauge="pin"
#   we subtract X[0,0] so the inter-block constraints are meaningful.
# ============================================================

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, Tuple


__all__ = ["VariationalLedger"]


class VariationalLedger(nn.Module):
    """
    Convex spacing energy on a ledger field X ∈ ℝ^{B×T}:

        E(X) =
          seat_weight  · Σ_b,t (X_{b,t+1} − X_{b,t} − Δ_b)^2
        + block_weight · Σ_b   (X_{b+1,0} − X_{b,0} − δ★)^2
        + tv_weight    · Σ_b,t |X_{b,t+1} − X_{b,t}|

    where Δ_b = δ★ / C_b and the last seat (t=C_b−1) uses the wrap target
    (Δ_b − δ★) against X_{b,0}.

    Parameters
    ----------
    delta : float
        Fundamental click size δ★.
    capacities : Iterable[int] | Tensor[int]
        Per-block seat capacities (C_b).
    tv_weight : float
        Total-variation penalty along seats (anti-jitter).
    seat_weight, block_weight : float
        Weights for seat- and block-spacing terms.
    gauge : {"pin","none"}
        If "pin", subtract X[0,0] to remove the global translation mode.
    """

    def __init__(
        self,
        delta,
        capacities,
        tv_weight: float = 0.0,
        seat_weight: float = 1.0,
        block_weight: float = 1.0,
        gauge: str = "pin",
    ):
        super().__init__()
        self.delta = float(delta)
        self.capacities = capacities  # kept as user-provided; normalized per-call
        self.tv_weight = float(tv_weight)
        self.seat_weight = float(seat_weight)
        self.block_weight = float(block_weight)
        if gauge not in ("pin", "none"):
            raise ValueError('gauge must be "pin" or "none"')
        self.gauge = gauge

    # ========================================================
    # Public API
    # ========================================================

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Alias for energy(X)."""
        return self.energy(X)

    def energy(self, X: torch.Tensor) -> torch.Tensor:
        """
        Full scalar energy: weighted sum of seat, block, and TV terms.
        """
        X = self._as_2d(X)
        X = self._gauge_fix(X)
        caps = self._expand_caps(X)
        parts = self.energy_parts(X, caps)
        return (
            self.seat_weight * parts["seat_term"]
            + self.block_weight * parts["block_term"]
            + self.tv_weight * parts["tv_term"]
        )

    def energy_parts(self, X: torch.Tensor, caps: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        """
        Return individual components {seat_term, block_term, tv_term} as scalars.
        """
        X = self._as_2d(X)
        X = self._gauge_fix(X)
        caps = self._expand_caps(X) if caps is None else caps

        # 1) Seat residuals (internal + wrap) with validity mask
        dif_seat, seat_mask = self._seat_residuals_with_wrap(X, caps)
        seat_term = (dif_seat.pow(2) * seat_mask).sum()

        # 2) Block residuals (between first seats of consecutive blocks)
        dif_block = self._block_residuals(X)
        block_term = dif_block.pow(2).sum()

        # 3) Total-variation penalty (valid internal edges only)
        if self.tv_weight > 0.0:
            tv_core = (X[:, 1:] - X[:, :-1]).abs()
            T = X.size(1)
            idx = torch.arange(T - 1, device=X.device).unsqueeze(0)  # (1,T-1)
            # valid if t < C_b - 1  ⇒ there exists an internal edge at (b,t→t+1)
            tv_mask = (idx < (caps.unsqueeze(1) - 1)).to(X.dtype)
            tv_term = (tv_core * tv_mask).sum()
        else:
            tv_term = X.new_zeros(())

        return {"seat_term": seat_term, "block_term": block_term, "tv_term": tv_term}

    def diagnostics(self, X: torch.Tensor) -> Dict[str, float]:
        """
        Normalized diagnostics: mean errors per relation for readability.
        """
        X = self._as_2d(X)
        X = self._gauge_fix(X)
        caps = self._expand_caps(X)
        parts = self.energy_parts(X, caps)

        B, _T = X.shape
        seat_edges = int(caps.sum().item())  # each valid seat contributes one residual
        seat_edges = max(seat_edges, 1)
        block_edges = max(B - 1, 1)

        seat_mse = (parts["seat_term"] / float(seat_edges)).item()
        block_mse = (parts["block_term"] / float(block_edges)).item()
        tv_mean = ((parts["tv_term"] / float(seat_edges)).item() if self.tv_weight > 0.0 else 0.0)

        return {
            "seat_mse": seat_mse,
            "block_mse": block_mse,
            "tv_mean": tv_mean,
            "B": int(B),
            "T": int(_T),
            "caps": self._expand_caps(X).detach().cpu().tolist(),
        }

    # ========================================================
    # Residual builders — equal-spacing relations
    # ========================================================

    def _seat_residuals_with_wrap(
        self, X: torch.Tensor, caps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        For block b and seat t, with Δ_b := δ★/C_b:

            internal edges (t < C_b−1):
                r_{b,t} = X_{b,t+1} − X_{b,t} − Δ_b

            wrap edge (t = C_b−1):
                r_{b,C_b−1} = X_{b,0} − X_{b,C_b−1} − (Δ_b − δ★)

        Returns:
            dif : (B,T) residuals (zeros where invalid)
            msk : (B,T) 1.0 for valid seats, else 0.0
        """
        B, T = X.shape
        dev, dt = X.device, X.dtype

        # Δ_b for each block, shape (B,1)
        step = (self.delta / caps.to(dt)).unsqueeze(1)

        dif = X.new_zeros(B, T)
        # internal edges where t ∈ [0, C_b−2]
        internal = X[:, 1:] - X[:, :-1] - step
        dif[:, :-1] = internal

        # wrap edge per block at index (C_b−1)
        last_idx = (caps - 1).clamp(min=0)            # (B,)
        rows = torch.arange(B, device=dev)
        x0 = X[:, :1]                                  # (B,1)
        xL = X.gather(1, last_idx.view(-1, 1))         # (B,1)
        wrap = x0 - xL - (step - self.delta)          # (B,1)
        dif[rows, last_idx] = wrap.squeeze(1)

        # mask of valid seats (t < C_b)
        idx = torch.arange(T, device=dev).unsqueeze(0)   # (1,T)
        mask = (idx < caps.unsqueeze(1)).to(dt)
        return dif, mask

    def _block_residuals(self, X: torch.Tensor) -> torch.Tensor:
        """Inter-block spacing residuals: X_{b+1,0} − X_{b,0} − δ★ for b=0..B-2."""
        if X.size(0) <= 1:
            return X.new_zeros((0,))
        first = X[:, 0]
        return (first[1:] - first[:-1]) - self.delta

    # ========================================================
    # Ideal layout generator (zero-residual layout up to gauge)
    # ========================================================

    def ideal_layout(
        self,
        B: int,
        T: int,
        caps: torch.Tensor | list | None = None,
        start: float = 0.0,
    ) -> torch.Tensor:
        """
        Construct X with zero residuals (modulo gauge):

            X_{b,t} = (start + b·δ★) + min(t, C_b−1) · (δ★/C_b)

        Seats with t ≥ C_b repeat the last value X_{b,C_b−1}.
        """
        device = torch.device("cpu")
        caps_t = torch.as_tensor(
            caps if caps is not None else self.capacities,
            dtype=torch.long,
            device=device,
        )
        if caps_t.numel() < B:
            caps_t = torch.cat([caps_t, caps_t.new_full((B - caps_t.numel(),), caps_t[-1].item())])
        elif caps_t.numel() > B:
            caps_t = caps_t[:B]
        caps_t = caps_t.clamp(min=1, max=T)

        # Vectorized construction
        b_idx = torch.arange(B, device=device, dtype=torch.float32).unsqueeze(1)  # (B,1)
        t_idx = torch.arange(T, device=device, dtype=torch.long).unsqueeze(0)     # (1,T)
        seat_idx = torch.minimum(t_idx.expand(B, T), (caps_t.unsqueeze(1) - 1))   # (B,T)
        step = (self.delta / caps_t.to(torch.float32)).unsqueeze(1)               # (B,1)
        base = (float(start) + b_idx * self.delta).to(torch.float32)              # (B,1)
        X = base + step * seat_idx.to(torch.float32)                              # (B,T)
        X = X.to(torch.float32)

        return self._gauge_fix(X) if self.gauge == "pin" else X

    # ========================================================
    # Internals
    # ========================================================

    @staticmethod
    def _as_2d(X: torch.Tensor) -> torch.Tensor:
        """Ensure shape (B,T)."""
        if X.dim() == 1:
            return X.unsqueeze(0)
        if X.dim() != 2:
            raise ValueError("X must be 1D or 2D (B,T)")
        return X

    def _gauge_fix(self, X: torch.Tensor) -> torch.Tensor:
        """Global pin (subtract X[0,0]) to remove translation ambiguity."""
        if self.gauge == "pin" and X.numel() > 0:
            return X - X[:1, :1]
        return X

    def _expand_caps(self, X: torch.Tensor) -> torch.Tensor:
        """
        Normalize `capacities` to a tensor of length B on X’s device, clamped to [1, T].
        If fewer than B are provided, the last capacity is repeated.
        """
        B, T = X.shape
        dev = X.device
        caps = torch.as_tensor(self.capacities, device=dev)
        if caps.numel() == 1:
            caps = caps.expand(B)
        elif caps.numel() < B:
            caps = torch.cat([caps, caps.new_full((B - caps.numel(),), caps[-1].item())])
        elif caps.numel() > B:
            caps = caps[:B]
        return caps.clamp(min=1, max=T).to(torch.long)
