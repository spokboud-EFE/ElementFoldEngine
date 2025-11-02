# ElementFold · domains/graph.py
# Irregular “graph” folds: non‑expansive diffusion operators on a normalized adjacency.
#
# Cookbook for non‑experts:
#   • normalize_adj(...)     → build Ā = D^{−1/2} A D^{−1/2}  (symmetric, with self‑loops by default)
#                              or row‑norm A_row = D^{−1} A   (set symmetric=False).
#   • PolyDiffusionFold      → y = Σ_{k=0..K} α_k · Ā^k x     (α ≥ 0, Σα = 1 ⇒ contraction on ‖·‖₂).
#   • ChebDiffusionFold      → y = Σ_{k=0..K} α_k · T_k(Ā) x  (Chebyshev T_k; also non‑expansive).
#   • make_fold(...)         → friendly factory returning a ready nn.Module ("poly" or "cheb").
#
# Shapes:
#   • x ∈ ℝ^{N×D} or ℝ^{B×N×D}. We treat the graph as shared across the batch.
#   • We use sparse SpMM (A@X) so large graphs remain memory‑light.
#
# Stability principle:
#   • Use a normalized adjacency with spectral radius ≤ 1 (symmetric Ā or row‑norm A_row).
#   • Mix powers (or Chebyshev polynomials) with convex weights α on the simplex.
#     Each term has operator norm ≤ 1 ⇒ convex mixture remains non‑expansive.

from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn


# ————————————————————————————————————————————————————————————
# Sparse helpers: build adjacency, coalesce duplicates, degree
# ————————————————————————————————————————————————————————————

@torch.no_grad()
def _coalesce(indices: torch.Tensor,
              values: torch.Tensor,
              size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Merge duplicate (i,j) entries by summing values.
    We briefly build a sparse COO, coalesce, and unpack.
    """
    A = torch.sparse_coo_tensor(indices, values, size=size).coalesce()
    return A.indices(), A.values()


@torch.no_grad()
def _build_adjacency(edge_index: torch.Tensor,
                     num_nodes: int,
                     edge_weight: Optional[torch.Tensor] = None,
                     symmetric: bool = True,
                     self_loops: bool = True,
                     self_loop_weight: float = 1.0,
                     dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """
    Build a raw adjacency A (COO form) with options:
      • symmetric=True  → add (j,i) for every (i,j) (undirected effect),
      • self_loops=True → add (i,i) with weight self_loop_weight (default 1.0).
    Returns (indices, values, size).
    """
    assert edge_index.dim() == 2 and edge_index.size(0) == 2, "edge_index must be (2,E)"
    E = edge_index.size(1)
    N = int(num_nodes)
    dev = edge_index.device

    if edge_weight is None:
        w = torch.ones(E, dtype=dtype, device=dev)
    else:
        w = edge_weight.to(device=dev, dtype=dtype)

    idx = edge_index.to(device=dev, dtype=torch.long)              # (2,E)

    if symmetric:
        rev = torch.stack([idx[1], idx[0]], dim=0)                 # (2,E)
        idx = torch.cat([idx, rev], dim=1)                         # (2,2E)
        w = torch.cat([w, w], dim=0)                               # (2E,)

    if self_loops:
        ii = torch.arange(N, device=dev, dtype=torch.long)
        self_idx = torch.stack([ii, ii], dim=0)                    # (2,N)
        self_w = torch.full((N,), float(self_loop_weight), device=dev, dtype=dtype)
        idx = torch.cat([idx, self_idx], dim=1)                    # (2,2E+N) or (2,E+N)
        w = torch.cat([w, self_w], dim=0)                          # (2E+N,) or (E+N,)

    idx, w = _coalesce(idx, w, size=(N, N))                        # merge duplicates
    return idx, w, (N, N)


@torch.no_grad()
def _degree_from_coo(indices: torch.Tensor,
                     values: torch.Tensor,
                     N: int,
                     row: bool = True) -> torch.Tensor:
    """
    Degree vector:
      • row=True  → out-degree: d[i] = Σ_j A_{i,j}
      • row=False → in-degree:  d[j] = Σ_i A_{i,j}
    """
    dev = values.device
    deg = torch.zeros(N, dtype=values.dtype, device=dev)
    sel = 0 if row else 1
    deg.index_add_(0, indices[sel], values)
    return deg.clamp_min(1e-12)                                    # avoid zeros for normalization


# ————————————————————————————————————————————————————————————
# Public: adjacency normalization
# ————————————————————————————————————————————————————————————

@torch.no_grad()
def normalize_adj(edge_index: torch.Tensor,
                  num_nodes: int,
                  edge_weight: Optional[torch.Tensor] = None,
                  *,
                  symmetric: bool = True,
                  self_loops: bool = True,
                  self_loop_weight: float = 1.0,
                  dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Build a normalized adjacency as a coalesced sparse_coo_tensor:

      • symmetric=True:
            Ā = D^{−1/2} A D^{−1/2}          (spectral radius ≤ 1 for undirected A + I)
        where d = row-degree of A (+ self-loops if enabled).

      • symmetric=False (row-normalized):
            A_row = D^{−1} A                 (row-stochastic; ρ(A_row) ≤ 1)
        where d = row-degree.

    Returns:
        torch.sparse_coo_tensor of shape (N,N), dtype=dtype, on the same device as edge_index.
    """
    idx, w, size = _build_adjacency(edge_index,
                                    num_nodes,
                                    edge_weight=edge_weight,
                                    symmetric=symmetric,           # mirror edges if requested
                                    self_loops=self_loops,
                                    self_loop_weight=self_loop_weight,
                                    dtype=dtype)
    N = size[0]
    d = _degree_from_coo(idx, w, N, row=True)                      # use row-degree for both modes

    if symmetric:
        d_inv_sqrt = d.rsqrt()                                     # D^{−1/2}
        scale = d_inv_sqrt.index_select(0, idx[0]) * d_inv_sqrt.index_select(0, idx[1])
        w_norm = w * scale
    else:
        d_inv = d.reciprocal()                                     # D^{−1}
        scale = d_inv.index_select(0, idx[0])
        w_norm = w * scale

    A_norm = torch.sparse_coo_tensor(idx, w_norm.to(dtype), size=size).coalesce()
    return A_norm


# ————————————————————————————————————————————————————————————
# Sparse matrix × dense matrix (SpMM) and a batched trick
# ————————————————————————————————————————————————————————————

def _spmm(A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """
    Y = A @ X for sparse A (N×N) and dense X (N×F).
    """
    assert A.is_sparse, "A must be a sparse COO tensor"
    assert X.dim() == 2 and A.size(0) == X.size(0), "X must be (N,F)"
    return torch.sparse.mm(A, X)


# ————————————————————————————————————————————————————————————
# Non‑expansive diffusion folds
# ————————————————————————————————————————————————————————————

class PolyDiffusionFold(nn.Module):
    """
    Non‑expansive polynomial diffusion on a normalized adjacency:

        y = Σ_{k=0..K} α_k · Ā^k x     with α_k ≥ 0 and Σ α_k = 1

    where Ā is either symmetric normalized (default) or row‑normalized.
    The convex simplex over α guarantees the operator has ‖·‖₂ ≤ 1.

    Input:
      x ∈ ℝ^{N×D} or ℝ^{B×N×D}
    """
    def __init__(self, A_hat: torch.Tensor, K: int = 3, learnable: bool = True):
        super().__init__()
        assert A_hat.is_sparse, "A_hat must be sparse COO (normalized)."
        self.register_buffer("A_hat", A_hat.coalesce(), persistent=False)   # normalized adjacency
        self._A_cache: Dict[torch.device, torch.Tensor] = {}               # per-device sparse cache
        self.K = int(max(0, K))
        # α logits initialized as δ at k=0 (α0≈1, others≈0) → identity at start
        init = torch.full((self.K + 1,), -9.0, dtype=torch.float32)
        init[0] = 9.0
        self.alpha_logits = nn.Parameter(init, requires_grad=bool(learnable))

    def _A_on(self, device: torch.device) -> torch.Tensor:
        A = self._A_cache.get(device)
        if A is None:
            A = self.A_hat.to(device).coalesce()
            self._A_cache[device] = A
        return A

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            N, D = x.shape
            X = x
            unflatten = lambda Y: Y
        elif x.dim() == 3:
            B, N, D = x.shape
            X = x.permute(1, 0, 2).reshape(N, B * D)                      # (N, B*D)
            unflatten = lambda Y: Y.view(N, B, D).permute(1, 0, 2)
        else:
            raise ValueError("x must be (N,D) or (B,N,D)")

        A = self._A_on(X.device)
        α = torch.softmax(self.alpha_logits, dim=0)                         # α ∈ Δ^{K}

        Yk = X                                                              # Y0 = X
        Y = α[0] * Yk
        for k in range(1, self.K + 1):
            Yk = _spmm(A, Yk)                                               # Yk = Ā · Y_{k-1}
            Y = Y + α[k] * Yk
        return unflatten(Y)


class ChebDiffusionFold(nn.Module):
    """
    Chebyshev diffusion with convex mixing:

        y = Σ_{k=0..K} α_k · T_k(Ā) x

    Chebyshev polynomials T_k on a matrix with ‖Ā‖₂ ≤ 1 satisfy ‖T_k(Ā)‖₂ ≤ 1.
    Convex mixing across k therefore remains non‑expansive.

    Recurrence:
        T0(X) = X
        T1(X) = Ā X
        T_{k}(X) = 2 Ā T_{k-1}(X) − T_{k-2}(X)
    """
    def __init__(self, A_hat: torch.Tensor, K: int = 3, learnable: bool = True):
        super().__init__()
        assert A_hat.is_sparse, "A_hat must be sparse COO (normalized)."
        self.register_buffer("A_hat", A_hat.coalesce(), persistent=False)
        self._A_cache: Dict[torch.device, torch.Tensor] = {}
        self.K = int(max(0, K))
        init = torch.full((self.K + 1,), -9.0, dtype=torch.float32)
        init[0] = 9.0
        self.alpha_logits = nn.Parameter(init, requires_grad=bool(learnable))

    def _A_on(self, device: torch.device) -> torch.Tensor:
        A = self._A_cache.get(device)
        if A is None:
            A = self.A_hat.to(device).coalesce()
            self._A_cache[device] = A
        return A

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            N, D = x.shape
            X = x
            unflatten = lambda Y: Y
        elif x.dim() == 3:
            B, N, D = x.shape
            X = x.permute(1, 0, 2).reshape(N, B * D)                      # (N, B*D)
            unflatten = lambda Y: Y.view(N, B, D).permute(1, 0, 2)
        else:
            raise ValueError("x must be (N,D) or (B,N,D)")

        A = self._A_on(X.device)
        α = torch.softmax(self.alpha_logits, dim=0)

        # T0(X) and, if needed, T1(X)
        T0 = X
        Y = α[0] * T0
        if self.K == 0:
            return unflatten(Y)

        T1 = _spmm(A, X)                                                   # Ā X
        Y = Y + α[1] * T1
        if self.K == 1:
            return unflatten(Y)

        # Recurrence for k ≥ 2
        Tkm2, Tkm1 = T0, T1
        for k in range(2, self.K + 1):
            Tk = 2.0 * _spmm(A, Tkm1) - Tkm2
            Y = Y + α[k] * Tk
            Tkm2, Tkm1 = Tkm1, Tk

        return unflatten(Y)


# ————————————————————————————————————————————————————————————
# Factory
# ————————————————————————————————————————————————————————————

def make_fold(num_nodes: int,
              edge_index: torch.Tensor,
              edge_weight: Optional[torch.Tensor] = None,
              *,
              kind: str = "poly",                 # 'poly' or 'cheb'
              K: int = 3,
              learnable: bool = True,
              symmetric: bool = True,
              self_loops: bool = True,
              self_loop_weight: float = 1.0,
              dtype: torch.dtype = torch.float32) -> nn.Module:
    """
    Build a graph fold operator in one call:
      1) construct normalized adjacency (symmetric Ā or row A_row),
      2) return a diffusion fold with convex α on the simplex.

    Args:
      num_nodes:        N
      edge_index:       (2,E) LongTensor
      edge_weight:      optional (E,) Tensor
      kind:             'poly' (powers) or 'cheb' (Chebyshev T_k)
      K:                polynomial order (0 ⇒ identity)
      learnable:        whether α logits are trainable
      symmetric:        use D^{−1/2} A D^{−1/2} if True; else row‑norm D^{−1} A
      self_loops:       add identity edges before normalization
      self_loop_weight: weight of self‑loops (default 1.0)
      dtype:            float32 by default

    Returns:
      nn.Module with forward(x) supporting (N,D) and (B,N,D).
    """
    A_norm = normalize_adj(edge_index=edge_index,
                           num_nodes=num_nodes,
                           edge_weight=edge_weight,
                           symmetric=symmetric,
                           self_loops=self_loops,
                           self_loop_weight=self_loop_weight,
                           dtype=dtype)
    if kind.lower() in ("poly", "polynomial"):
        return PolyDiffusionFold(A_norm, K=K, learnable=learnable)
    if kind.lower() in ("cheb", "chebyshev"):
        return ChebDiffusionFold(A_norm, K=K, learnable=learnable)
    raise ValueError(f"unknown fold kind: {kind!r} (expected 'poly' or 'cheb')")


# ————————————————————————————————————————————————————————————
# Utilities for debugging/inspection
# ————————————————————————————————————————————————————————————

def to_dense(A: torch.Tensor) -> torch.Tensor:
    """Convert a sparse COO adjacency to a dense (N,N) tensor."""
    assert A.is_sparse, "A must be sparse COO"
    return A.to_dense()


__all__ = [
    "normalize_adj",
    "PolyDiffusionFold",
    "ChebDiffusionFold",
    "make_fold",
    "to_dense",
]
