# ElementFold · variational.py
# This file defines the convex “tension” that keeps the ledger equally spaced.
# It penalizes two kinds of misalignment:
#   1) seats inside a block drifting away from δ⋆/C spacing,
#   2) consecutive blocks drifting away from δ⋆ offset.
# Optionally, a small total‑variation (TV) term discourages jitter.

import torch, torch.nn as nn  # We use PyTorch tensors and nn.Module to plug into the engine cleanly.


class VariationalLedger(nn.Module):  # A Module so it can move to GPU, save, and compose with other parts.
    def __init__(
        self,
        delta,                       # δ⋆: the fundamental “click” size.
        capacities,                  # C per block: how many seats live inside one click in that block.
        tv_weight=0.0,               # Weight on the total‑variation penalty along seats.
        seat_weight=1.0,             # Weight for the seat‑spacing penalty (default 1).
        block_weight=1.0,            # Weight for the block‑spacing penalty (default 1).
        gauge="pin",                 # How to remove the global shift ambiguity (“gauge”): 'pin' or 'none'.
    ):
        super().__init__()           # Standard nn.Module initialization.
        self.delta = float(delta)    # Store δ⋆ as a plain float for speed and clarity.
        self.capacities = capacities # Keep raw user input; we’ll normalize it later per batch.
        self.tv_weight = float(tv_weight)      # Cast weights to float for stable arithmetic.
        self.seat_weight = float(seat_weight)  # Seat penalty scale.
        self.block_weight = float(block_weight)# Block penalty scale.
        self.gauge = str(gauge)                # Gauge mode as string.

    # ————————————————————————————————————————————————
    # Public API expected by the rest of the engine
    # ————————————————————————————————————————————————

    def forward(self, X):            # Calling the module returns the scalar energy.
        return self.energy(X)        # We defer to the explicit 'energy' method for readability.

    def energy(self, X):             # Full scalar energy = seat_term + block_term + tv_term (with weights).
        X = self._as_2d(X)           # Ensure X has shape (B,T): B blocks (rows), T seats (cols).
        X = self._gauge_fix(X)       # Remove the irrelevant global shift if requested (stabilizes optimization).

        caps = self._expand_caps(X)  # Produce a length‑B tensor of capacities, each clamped to [1, T].
        parts = self.energy_parts(X, caps)                # Compute individual components as a dict.
        # Combine pieces with their weights into one scalar; '.sum()' ensures a 0‑D tensor even if any piece is empty.
        total = (
            self.seat_weight * parts["seat_term"]
            + self.block_weight * parts["block_term"]
            + self.tv_weight * parts["tv_term"]
        )
        return total  # A single scalar tensor (0‑D) that autograd can differentiate.

    # ————————————————————————————————————————————————
    # Detailed components and diagnostics
    # ————————————————————————————————————————————————

    def energy_parts(self, X, caps=None):   # Return a dict with each penalty so callers can inspect them.
        X = self._as_2d(X)                  # Make sure X is (B,T).
        X = self._gauge_fix(X)              # Apply gauge pin if enabled.
        caps = self._expand_caps(X) if caps is None else caps  # Ensure capacities are present and valid.

        dif_seat, seat_mask = self.seat_residuals(X, caps)     # Seat residuals: (X_{t+1} − X_t − δ⋆/C_b).
        seat_term = (dif_seat.pow(2) * seat_mask).sum()        # Sum of squares over valid seats only.

        dif_block = self.block_residuals(X)                    # Block residuals: (X_{b+1,0} − X_{b,0} − δ⋆).
        block_term = dif_block.pow(2).sum()                    # Sum of squares across consecutive blocks.

        if self.tv_weight > 0.0:                               # If TV is active, compute it; otherwise return zero.
            tv_core = (X[:, 1:] - X[:, :-1]).abs()             # Absolute first difference along seats.
            # TV should only act on seats that exist in that block; build a mask for columns 1..T−1.
            T = X.size(1)                                      # Number of seat columns.
            idx = torch.arange(T - 1, device=X.device).unsqueeze(0)           # Column indices for TV pairs.
            tv_mask = (idx < (caps.unsqueeze(1) - 1)).to(X.dtype)             # Mask valid seat differences.
            tv_term = (tv_core * tv_mask).sum()                # Sum absolute differences over valid seats.
        else:
            tv_term = X.new_zeros(())                          # Scalar zero (same device/dtype as X).

        return {"seat_term": seat_term, "block_term": block_term, "tv_term": tv_term}  # Dictionary of parts.

    def diagnostics(self, X):                                   # Provide normalized, human‑interpretable diagnostics.
        X = self._as_2d(X)                                      # Ensure shape (B,T).
        caps = self._expand_caps(X)                             # One capacity per block.
        parts = self.energy_parts(X, caps)                      # Compute raw components.

        # Count how many seat relations we actually enforced: for each block, there are C_b seat edges in the cycle.
        seat_edges = caps.to(X.dtype).sum().clamp_min(1)        # Avoid division by zero just in case.
        # For blocks, there are B−1 relations (between consecutive blocks); clamp at 1 to avoid div by zero when B==1.
        block_edges = max(int(X.size(0)) - 1, 0) or 1

        # Normalize terms to a per‑edge mean so they are comparable across shapes and capacities.
        seat_mse = (parts["seat_term"] / seat_edges).item()     # Average squared seat error per seat relation.
        block_mse = (parts["block_term"] / block_edges).item()  # Average squared block error per block relation.
        tv_mean = (parts["tv_term"] / seat_edges).item() if self.tv_weight > 0.0 else 0.0  # Mean TV per seat.

        return {
            "seat_mse": seat_mse,                 # How far seats deviate from perfect δ⋆/C spacing on average.
            "block_mse": block_mse,               # How far blocks deviate from perfect δ⋆ jumps on average.
            "tv_mean": tv_mean,                   # Average absolute seat difference (if enabled).
            "B": int(X.size(0)),                  # Number of blocks.
            "T": int(X.size(1)),                  # Number of seats per block (max columns provided).
            "caps": caps.detach().cpu().tolist(), # The capacities used per block, for transparency.
        }

    # ————————————————————————————————————————————————
    # Residual builders (these express the exact equal‑spacing equations)
    # ————————————————————————————————————————————————

    def seat_residuals(self, X, caps=None):        # For each block b and seat t: r_b,t = X_{b,t+1} − X_{b,t} − δ⋆/C_b.
        X = self._as_2d(X)                         # Ensure 2D shape.
        caps = self._expand_caps(X) if caps is None else caps  # One capacity per block.
        B, T = X.shape                              # Dimensions for convenience.
        step = (self.delta / caps.to(X.dtype)).unsqueeze(1)     # Per‑block target step Δ_b = δ⋆/C_b, broadcast along seats.

        # Build a mask indicating which seat columns are valid per block: columns 0..C_b−1 are valid.
        idx = torch.arange(T, device=X.device).unsqueeze(0)     # Column index grid (1×T).
        seat_mask = (idx < caps.unsqueeze(1)).to(X.dtype)       # Mask 1.0 where seat exists, 0.0 otherwise.

        # For the cyclic seat equalities, we “roll” by −1 so col t+1 aligns with col t, then subtract the target step.
        dif = X.roll(shifts=-1, dims=1) - X - step              # Residuals of equal‑spacing constraints.

        return dif, seat_mask                                    # Caller multiplies by mask before summing.

    def block_residuals(self, X):                      # For consecutive blocks: r_b = X_{b+1,0} − X_{b,0} − δ⋆.
        X = self._as_2d(X)                             # Ensure 2D shape.
        if X.size(0) <= 1:                             # If only one block, there is no block relation to enforce.
            return X.new_zeros((0,))                   # Return an empty vector (shapes nicely in sums).
        b0 = X[:, 0]                                   # First seat from each block, shape (B,).
        return (b0[1:] - b0[:-1]) - self.delta         # Differences between neighbors minus δ⋆.

    # ————————————————————————————————————————————————
    # Construct an “ideal” ledger (useful for warm‑starts or assertions)
    # ————————————————————————————————————————————————

    def ideal_layout(self, B, T, caps=None, start=0.0):   # Build X where all residuals are exactly zero (up to gauge).
        # Create an empty (B,T) tensor to fill with ideal positions.
        dev = torch.device("cpu")                         # Default device; caller can .to(device) after if needed.
        X = torch.empty(B, T, dtype=torch.float32, device=dev)  # Allocate the ledger grid.

        # Expand capacities to length‑B and clamp to [1, T] so we never step outside columns.
        caps = torch.as_tensor(caps if caps is not None else self.capacities, dtype=torch.long, device=dev)
        if caps.numel() < B:                              # If fewer than B capacities were provided, repeat the last one.
            last = caps[-1].item()
            caps = torch.cat([caps, caps.new_full((B - caps.numel(),), last)], dim=0)
        caps = caps.clamp(min=1, max=T)                   # Make sure every capacity is valid.

        # For each block b, fill T columns so that:
        #   (i) seats 0..C_b−1 are spaced by δ⋆/C_b inside the block,
        #  (ii) block b starts at 'start + b·δ⋆' (so inter‑block relation is exactly δ⋆).
        for b in range(B):                                 # Loop over blocks for clarity (B is small in practice).
            Cb = max(1, int(caps[b].item()))              # Capacity of this block as a plain int.
            step = self.delta / Cb                         # Seat spacing inside this block.
            base = start + b * self.delta                  # Starting position for block b so that blocks are δ⋆ apart.
            # Fill the first Cb seats with exact arithmetic progression; leave remaining seats (if any) as copies of last.
            for t in range(T):                             # Loop over seats in this row.
                X[b, t] = base + step * min(t, Cb - 1)    # After the last valid seat, we keep the edge value (harmless for masking).
        # Apply gauge fix to align with the rest of this class if needed.
        return self._gauge_fix(X) if self.gauge == "pin" else X

    # ————————————————————————————————————————————————
    # Internal helpers
    # ————————————————————————————————————————————————

    def _as_2d(self, X):                        # Accept a 1D vector for convenience and promote to (1,T).
        return X.unsqueeze(0) if X.dim() == 1 else X

    def _gauge_fix(self, X):                    # Remove the global translation ambiguity to stabilize optimization.
        if self.gauge == "pin":                 # “Pin” the first seat of each block to zero by subtracting it.
            return X - X[:, :1]                 # This keeps differences intact but avoids drifting means.
        return X                                # If gauge is 'none', leave X as is.

    def _expand_caps(self, X):                  # Normalize the capacities input into a tensor of length B on X's device/dtype.
        B, T = X.shape                          # Read batch dimensions.
        dev, dt = X.device, X.dtype             # Cache device and dtype for new tensors.
        caps_raw = self.capacities              # Whatever the user passed at construction time.

        caps = torch.as_tensor(caps_raw, device=dev)      # Convert to a tensor on the same device.
        if caps.numel() == 1:                               # If a single capacity was provided, share it across blocks.
            caps = caps.expand(B)
        elif caps.numel() < B:                              # If fewer than B capacities, repeat the last for the rest.
            last = caps[-1].item()
            caps = torch.cat([caps, caps.new_full((B - caps.numel(),), last)], dim=0)
        elif caps.numel() > B:                              # If more than B provided, truncate to B.
            caps = caps[:B]

        caps = caps.clamp(min=1, max=T).to(torch.long)      # Ensure each capacity is within [1, T] and integer.
        return caps                                         # Return per‑block capacities ready for masking/steps.
