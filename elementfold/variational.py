# ElementFold · variational.py
# ============================================================
# Variational Ledger — convex energy that keeps the ledger evenly spaced.
#
# Physical intuition
# ------------------
# • Each block (a click, width δ★) contains C_b seats.
# • Inside a block: seat spacing should be Δ_b = δ★ / C_b.
# • Across blocks: first seat of block b+1 should advance by exactly δ★.
# • The last seat wraps back to the first seat, so the wrap edge target is
#       (δ★/C_b − δ★)
#   instead of δ★/C_b — a crucial correction for coherence.
#
# Gauge choice:
#   A global offset X → X + const changes nothing physically.
#   We fix (“pin”) one global origin X[0,0] so inter-block constraints stay meaningful.
# ============================================================

from __future__ import annotations
import torch
import torch.nn as nn


class VariationalLedger(nn.Module):
    """
    Convex spacing energy on the ledger:

        E = w_seat  Σ_b,t (X_{b,t+1} − X_{b,t} − Δ_b,t)^2
          + w_block Σ_b   (X_{b+1,0} − X_{b,0} − δ★)^2
          + w_tv    Σ_b,t |X_{b,t+1} − X_{b,t}|

    Parameters
    ----------
    delta : float
        Fundamental click size δ★.
    capacities : iterable
        Per-block seat capacities (C_b).
    tv_weight : float
        Total-variation regularizer along seats (anti-jitter).
    seat_weight, block_weight : float
        Relative weights of seat and block spacing terms.
    gauge : {'pin','none'}
        'pin' → subtract global reference X[0,0].
    """

    def __init__(self, delta, capacities,
                 tv_weight: float = 0.0,
                 seat_weight: float = 1.0,
                 block_weight: float = 1.0,
                 gauge: str = "pin"):
        super().__init__()
        self.delta = float(delta)
        self.capacities = capacities
        self.tv_weight = float(tv_weight)
        self.seat_weight = float(seat_weight)
        self.block_weight = float(block_weight)
        self.gauge = str(gauge)

    # ========================================================
    # Public API
    # ========================================================

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Alias for energy(X)."""
        return self.energy(X)

    def energy(self, X: torch.Tensor) -> torch.Tensor:
        """Full scalar energy = weighted sum of seat, block, and TV terms."""
        X = self._as_2d(X)
        X = self._gauge_fix(X)
        caps = self._expand_caps(X)
        parts = self.energy_parts(X, caps)
        return (
            self.seat_weight  * parts["seat_term"]
          + self.block_weight * parts["block_term"]
          + self.tv_weight    * parts["tv_term"]
        )

    def energy_parts(self, X: torch.Tensor, caps: torch.Tensor | None = None) -> dict:
        """Return individual components {seat_term, block_term, tv_term}."""
        X = self._as_2d(X)
        X = self._gauge_fix(X)
        caps = self._expand_caps(X) if caps is None else caps

        # 1) Seat residuals (internal + wrap)
        dif_seat, seat_mask = self._seat_residuals_with_wrap(X, caps)
        seat_term = (dif_seat.pow(2) * seat_mask).sum()

        # 2) Block residuals (between first seats)
        dif_block = self._block_residuals(X)
        block_term = dif_block.pow(2).sum()

        # 3) Optional total-variation penalty
        if self.tv_weight > 0.0:
            tv_core = (X[:, 1:] - X[:, :-1]).abs()
            T = X.size(1)
            idx = torch.arange(T - 1, device=X.device).unsqueeze(0)
            tv_mask = (idx < (caps.unsqueeze(1) - 1)).to(X.dtype)
            tv_term = (tv_core * tv_mask).sum()
        else:
            tv_term = X.new_zeros(())

        return {"seat_term": seat_term, "block_term": block_term, "tv_term": tv_term}

    def diagnostics(self, X: torch.Tensor) -> dict:
        """Readable, normalized diagnostics (mean squared errors per relation)."""
        X = self._as_2d(X)
        X = self._gauge_fix(X)
        caps = self._expand_caps(X)
        parts = self.energy_parts(X, caps)

        seat_edges = caps.to(X.dtype).sum().clamp_min(1)
        block_edges = max(int(X.size(0)) - 1, 1)
        seat_mse  = (parts["seat_term"]  / seat_edges).item()
        block_mse = (parts["block_term"] / block_edges).item()
        tv_mean   = (parts["tv_term"]    / seat_edges).item() if self.tv_weight > 0 else 0.0

        return {
            "seat_mse":  seat_mse,
            "block_mse": block_mse,
            "tv_mean":   tv_mean,
            "B": int(X.size(0)), "T": int(X.size(1)),
            "caps": self._expand_caps(X).detach().cpu().tolist(),
        }

    # ========================================================
    # Residual builders — equal-spacing equations
    # ========================================================

    def _seat_residuals_with_wrap(self, X: torch.Tensor, caps: torch.Tensor
                                  ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        For block b and seat t:

            r_b,t = X_{b,t+1} − X_{b,t} − Δ_b       (internal edges)
            r_b,Cb−1 = X_{b,0} − X_{b,Cb−1} − (Δ_b − δ★)  (wrap edge)

        Returns residuals (B,T) and validity mask (B,T).
        """
        B, T = X.shape
        dev, dt = X.device, X.dtype
        step = (self.delta / caps.to(dt)).unsqueeze(1)
        dif = X.new_zeros(B, T)

        # Internal edges
        dnext = X[:, 1:] - X[:, :-1] - step
        dif[:, :-1] = dnext

        # Wrap edge
        last_idx = (caps - 1).clamp(min=0)
        rows = torch.arange(B, device=dev)
        x0 = X[:, :1]
        xL = X.gather(1, last_idx.view(-1, 1))
        wrap = x0 - xL - (step - self.delta)
        dif[rows, last_idx] = wrap.squeeze(1)

        # Mask: valid seats only
        idx = torch.arange(T, device=dev).unsqueeze(0)
        mask = (idx < caps.unsqueeze(1)).to(dt)
        return dif, mask

    def _block_residuals(self, X: torch.Tensor) -> torch.Tensor:
        """Inter-block spacing residuals: X_{b+1,0} − X_{b,0} − δ★."""
        if X.size(0) <= 1:
            return X.new_zeros((0,))
        b0 = X[:, 0]
        return (b0[1:] - b0[:-1]) - self.delta

    # ========================================================
    # Ideal layout generator (for checks or warm starts)
    # ========================================================

    def ideal_layout(self, B: int, T: int,
                     caps: torch.Tensor | list | None = None,
                     start: float = 0.0) -> torch.Tensor:
        """
        Produce X with zero residuals (up to gauge):
            X_{b,t} = (start + b·δ★) + t·(δ★/C_b)
        Seats ≥ C_b repeat the last value.
        """
        dev = torch.device("cpu")
        X = torch.empty(B, T, dtype=torch.float32, device=dev)
        caps = torch.as_tensor(caps if caps is not None else self.capacities,
                               dtype=torch.long, device=dev)
        if caps.numel() < B:
            caps = torch.cat([caps,
                              caps.new_full((B - caps.numel(),), caps[-1].item())])
        caps = caps.clamp(min=1, max=T)

        for b in range(B):
            Cb = int(caps[b].item())
            step = self.delta / max(1, Cb)
            base = start + b * self.delta
            for t in range(T):
                X[b, t] = base + step * min(t, Cb - 1)
        return self._gauge_fix(X) if self.gauge == "pin" else X

    # ========================================================
    # Internals
    # ========================================================

    def _as_2d(self, X: torch.Tensor) -> torch.Tensor:
        """Ensure shape (B,T)."""
        return X.unsqueeze(0) if X.dim() == 1 else X

    def _gauge_fix(self, X: torch.Tensor) -> torch.Tensor:
        """Global pin (subtract X[0,0]) to remove translation ambiguity."""
        if self.gauge == "pin" and X.numel() > 0:
            return X - X[:1, :1]
        return X

    def _expand_caps(self, X: torch.Tensor) -> torch.Tensor:
        """Normalize `capacities` → tensor length B on X’s device, clamped [1,T]."""
        B, T = X.shape
        dev = X.device
        caps = torch.as_tensor(self.capacities, device=dev)
        if caps.numel() == 1:
            caps = caps.expand(B)
        elif caps.numel() < B:
            caps = torch.cat([caps, caps.new_full((B - caps.numel(),),
                                                  caps[-1].item())])
        elif caps.numel() > B:
            caps = caps[:B]
        return caps.clamp(min=1, max=T).to(torch.long)
