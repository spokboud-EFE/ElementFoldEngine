# ElementFold · variational.py
# ──────────────────────────────────────────────────────────────────────────────
# Intuition (plain words)
# -----------------------
# The ledger wants to be “evenly toothed”:
#   • inside each block (one click), seats should be spaced by Δ_b = δ⋆ / C_b,
#   • across blocks, the first seat should advance by exactly δ⋆,
#   • (optionally) avoid jitter with a small TV penalty along seats.
#
# Subtleties we handle explicitly:
#   • The last seat wraps back to the first seat across the click boundary.
#     The correct target difference for that wrap edge is (δ⋆/C_b − δ⋆),
#     not δ⋆/C_b. This matters a lot for coherence.
#   • Gauge: there is a global translation ambiguity X → X + const.
#     We “pin” a **single global** reference (X[0,0]) so inter‑block spacing
#     constraints stay meaningful (we do NOT pin each row independently).

from __future__ import annotations
import torch
import torch.nn as nn


class VariationalLedger(nn.Module):
    """
    Convex tension on the ledger:
      E = w_seat · Σ_b,t   (X_{b,t+1} − X_{b,t} − target_b,t)^2
        + w_block · Σ_b    (X_{b+1,0} − X_{b,0} − δ⋆)^2
        + w_tv · Σ_b,t     |X_{b,t+1} − X_{b,t}|   (masked to valid seats)

    Public API (used by train.py):
        var = VariationalLedger(delta, capacities, tv_weight).to(device)
        e   = var.energy(X[:, :maxcap])
    """

    def __init__(
        self,
        delta,                       # δ⋆: fundamental click size.
        capacities,                  # per‑block seat capacities C_b (scalar, list/1D tensor, or broadcastable).
        tv_weight: float = 0.0,      # weight for total‑variation regularizer along seats.
        seat_weight: float = 1.0,    # weight for seat‑spacing penalty.
        block_weight: float = 1.0,   # weight for block‑spacing penalty.
        gauge: str = "pin",          # 'pin' = subtract X[0,0] globally; 'none' = no gauge fix.
    ):
        super().__init__()
        self.delta = float(delta)
        self.capacities = capacities
        self.tv_weight = float(tv_weight)
        self.seat_weight = float(seat_weight)
        self.block_weight = float(block_weight)
        self.gauge = str(gauge)

    # ————————————————————————————————————————————————
    # Public API
    # ————————————————————————————————————————————————

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.energy(X)

    def energy(self, X: torch.Tensor) -> torch.Tensor:
        """
        Full scalar energy = seat_term + block_term + tv_term (with weights).
        """
        X = self._as_2d(X)
        X = self._gauge_fix(X)

        caps = self._expand_caps(X)               # (B,) capacities clamped to [1, T]
        parts = self.energy_parts(X, caps)        # compute individual components
        total = (
            self.seat_weight  * parts["seat_term"] +
            self.block_weight * parts["block_term"] +
            self.tv_weight    * parts["tv_term"]
        )
        return total  # 0‑D tensor

    def energy_parts(self, X: torch.Tensor, caps: torch.Tensor | None = None) -> dict:
        """
        Return a dict with the unweighted components {seat_term, block_term, tv_term}.
        """
        X = self._as_2d(X)
        X = self._gauge_fix(X)
        caps = self._expand_caps(X) if caps is None else caps

        dif_seat, seat_mask = self._seat_residuals_with_wrap(X, caps)   # (B,T) each
        seat_term = (dif_seat.pow(2) * seat_mask).sum()

        dif_block = self._block_residuals(X)                             # (B−1,)
        block_term = dif_block.pow(2).sum()

        if self.tv_weight > 0.0:
            tv_core = (X[:, 1:] - X[:, :-1]).abs()
            T = X.size(1)
            idx = torch.arange(T - 1, device=X.device).unsqueeze(0)     # (1, T−1)
            tv_mask = (idx < (caps.unsqueeze(1) - 1)).to(X.dtype)       # valid along seats 0..C_b−2
            tv_term = (tv_core * tv_mask).sum()
        else:
            tv_term = X.new_zeros(())

        return {"seat_term": seat_term, "block_term": block_term, "tv_term": tv_term}

    def diagnostics(self, X: torch.Tensor) -> dict:
        """
        Human‑readable, normalized diagnostics (means per enforced relation).
        """
        X = self._as_2d(X)
        X = self._gauge_fix(X)
        caps = self._expand_caps(X)
        parts = self.energy_parts(X, caps)

        seat_edges = caps.to(X.dtype).sum().clamp_min(1)   # for each block: (C_b − 1) + 1wrap = C_b constraints
        block_edges = max(int(X.size(0)) - 1, 0) or 1

        seat_mse  = (parts["seat_term"]  / seat_edges).item()
        block_mse = (parts["block_term"] / block_edges).item()
        tv_mean   = (parts["tv_term"]    / seat_edges).item() if self.tv_weight > 0.0 else 0.0

        return {
            "seat_mse":  seat_mse,
            "block_mse": block_mse,
            "tv_mean":   tv_mean,
            "B":         int(X.size(0)),
            "T":         int(X.size(1)),
            "caps":      self._expand_caps(X).detach().cpu().tolist(),
        }

    # ————————————————————————————————————————————————
    # Residual builders (exact equal‑spacing equations)
    # ————————————————————————————————————————————————

    def _seat_residuals_with_wrap(self, X: torch.Tensor, caps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        For each block b and seat t we form residuals:

            r_b,t = X_{b,t+1} − X_{b,t} − Δ_b      for t = 0..C_b−2     (internal edges)
            r_b,C_b−1 = X_{b,0} − X_{b,C_b−1} − (Δ_b − δ⋆)             (wrap edge)

        Everything outside 0..C_b−1 is masked out.
        Returns:
            dif   : (B,T) residuals (zeros where masked)
            mask  : (B,T) 1.0 for valid columns t< C_b, else 0.0
        """
        B, T = X.shape
        dev, dt = X.device, X.dtype

        # Per‑block seat step Δ_b
        step = (self.delta / caps.to(dt)).unsqueeze(1)         # (B,1)

        # Start with all‑zeros residual grid
        dif = X.new_zeros(B, T)

        # Internal seat differences (t = 0..T−2); we will mask to t < C_b − 1
        dnext = X[:, 1:] - X[:, :-1] - step                    # (B, T−1)
        dif[:, :-1] = dnext

        # Wrap residual at column (C_b − 1) for each block: X_{b,0} − X_{b,C_b−1} − (Δ_b − δ⋆)
        last_idx = (caps - 1).clamp(min=0)                     # (B,)
        rows = torch.arange(B, device=dev)
        x0   = X[:, :1]                                        # (B,1)
        xL   = X.gather(1, last_idx.view(-1, 1))               # (B,1)
        wrap = x0 - xL - (step - self.delta)                   # (B,1)
        dif[rows, last_idx] = wrap.squeeze(1)                  # place at column C_b−1 per row

        # Seat mask: valid columns 0..C_b−1
        idx = torch.arange(T, device=dev).unsqueeze(0)         # (1, T)
        mask = (idx < caps.unsqueeze(1)).to(dt)                # (B, T)

        return dif, mask

    def _block_residuals(self, X: torch.Tensor) -> torch.Tensor:
        """
        Consecutive block spacing:
            r_b = X_{b+1,0} − X_{b,0} − δ⋆   for b = 0..B−2
        """
        X = self._as_2d(X)
        if X.size(0) <= 1:
            return X.new_zeros((0,))
        b0 = X[:, 0]
        return (b0[1:] - b0[:-1]) - self.delta

    # ————————————————————————————————————————————————
    # Ideal layout (useful for warm‑starts or assertions)
    # ————————————————————————————————————————————————

    def ideal_layout(self, B: int, T: int, caps: torch.Tensor | list | None = None, start: float = 0.0) -> torch.Tensor:
        """
        Construct X where all residuals are zero (up to the chosen gauge):
            X_{b,t} = (start + b·δ⋆) + t·(δ⋆/C_b)   for t = 0..C_b−1   (then hold last value for columns ≥ C_b)
        """
        dev = torch.device("cpu")
        X = torch.empty(B, T, dtype=torch.float32, device=dev)

        caps = torch.as_tensor(caps if caps is not None else self.capacities, dtype=torch.long, device=dev)
        if caps.numel() < B:
            caps = torch.cat([caps, caps.new_full((B - caps.numel(),), caps[-1].item())], dim=0)
        caps = caps.clamp(min=1, max=T)

        for b in range(B):
            Cb = int(caps[b].item())
            step = self.delta / max(1, Cb)
            base = start + b * self.delta
            for t in range(T):
                X[b, t] = base + step * min(t, Cb - 1)

        return self._gauge_fix(X) if self.gauge == "pin" else X

    # ————————————————————————————————————————————————
    # Internals
    # ————————————————————————————————————————————————

    def _as_2d(self, X: torch.Tensor) -> torch.Tensor:
        return X.unsqueeze(0) if X.dim() == 1 else X

    def _gauge_fix(self, X: torch.Tensor) -> torch.Tensor:
        """
        Global pin: subtract X[0,0] (a single scalar) from all entries.
        Preserves inter‑block differences and makes block terms meaningful.
        """
        if self.gauge == "pin" and X.numel() > 0:
            return X - X[:1, :1]   # broadcasts (1,1) → (B,T)
        return X

    def _expand_caps(self, X: torch.Tensor) -> torch.Tensor:
        """
        Normalize the constructor's `capacities` into a length‑B integer tensor
        on X’s device, clamped to [1, T].
        """
        B, T = X.shape
        dev = X.device
        caps = torch.as_tensor(self.capacities, device=dev)

        if caps.numel() == 1:
            caps = caps.expand(B)
        elif caps.numel() < B:
            caps = torch.cat([caps, caps.new_full((B - caps.numel(),), caps[-1].item())], dim=0)
        elif caps.numel() > B:
            caps = caps[:B]

        return caps.clamp(min=1, max=T).to(torch.long)
