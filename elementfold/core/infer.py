# ElementFold · infer.py
# ============================================================
# Inference — calm twin of training.
#
# Purpose
# -------
#   • Runs a model forward pass (logits, ledger X).
#   • Converts logits → tokens using greedy or stochastic sampling.
#   • Optionally applies a relaxation clock (fold counter ℱ, smoothing step).
#
# Design
# -------
#   Deterministic by default (greedy).  Add `strategy='sample'` to explore:
#     temperature scaling, top-k, top-p (nucleus) sampling.
#   Relaxation dictionary lets the ledger “breathe”:
#       ℱ accumulates small shares η·(1 + w·|ΔX|),
#       and one optional Euler step applies −λ(Φ−Φ_∞) + D∇²Φ.
# ============================================================

from __future__ import annotations
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn.functional as F

__all__ = ["infer_loop"]

# ============================================================
# 1) Sampling utilities
# ============================================================

def _apply_topk(logits: torch.Tensor, k: Optional[int]) -> torch.Tensor:
    """Keep only the top-k logits per position; others → −∞ (no change if k None/≤0/≥V)."""
    if not k or k <= 0:
        return logits
    V = logits.size(-1)
    k = int(k)
    if k >= V:
        return logits
    vals, idx = torch.topk(logits, k, dim=-1)
    pruned = torch.full_like(logits, float("-inf"))
    return pruned.scatter(-1, idx, vals)


def _apply_topp(logits: torch.Tensor, p: Optional[float]) -> torch.Tensor:
    """Nucleus (top-p) filtering — retain minimal prefix with cumulative prob ≥ p (always keep argmax)."""
    if p is None:
        return logits
    p = float(p)
    if not (0.0 < p < 1.0):
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum = probs.cumsum(dim=-1)
    mask = cum > p
    mask[..., 0] = False  # always keep best
    kept = sorted_logits.masked_fill(mask, float("-inf"))
    out = torch.full_like(logits, float("-inf"))
    return out.scatter(-1, sorted_idx, kept)


def _divide_by_temperature(
    logits: torch.Tensor,
    temperature: float | torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """Divide logits by temperature (supports scalar or per-position tensors)."""
    if torch.is_tensor(temperature):
        T = temperature.to(dtype=logits.dtype, device=logits.device).clamp_min(eps)
        if T.dim() == 2:  # (B,T) → (B,T,1)
            T = T.unsqueeze(-1)
        return logits / T
    return logits / max(float(temperature), eps)


def _sample_from_logits(
    logits: torch.Tensor,
    *,
    temperature: float | torch.Tensor = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Convert logits → token ids using temperature scaling + top-k/top-p.
    Deterministic if temperature ≤ 0 (argmax).
    """
    # Deterministic case
    if (not torch.is_tensor(temperature)) and float(temperature) <= 0:
        return logits.argmax(dim=-1)

    # Scale + filter
    logits = _divide_by_temperature(logits, temperature)
    logits = _apply_topk(logits, top_k)
    logits = _apply_topp(logits, top_p)

    # Softmax with NaN guard (can occur if all entries are -inf after filtering)
    probs = torch.softmax(logits, dim=-1)
    B, T, V = probs.shape
    flat = probs.reshape(B * T, V)
    bad = torch.isnan(flat).any(dim=1)
    if bad.any():
        flat[bad] = 1.0 / V

    idx = torch.multinomial(flat, 1, generator=generator)
    return idx.view(B, T)


# ============================================================
# 2) Relaxation helpers (optional)
# ============================================================

def _fold_counter(X: torch.Tensor, *, eta: float, eta_path_weight: float) -> torch.Tensor:
    """Compute fold counter ℱ:  ℱ_t = Σ_t η·(1 + w·|ΔX|), nonnegative and cumulative along T."""
    if eta <= 0 and eta_path_weight <= 0:
        return torch.zeros_like(X)
    dX = torch.zeros_like(X)
    dX[:, 1:] = (X[:, 1:] - X[:, :-1]).abs()
    dF = eta * (1.0 + eta_path_weight * dX)
    F = torch.cumsum(dF, dim=1)
    return torch.clamp(F, min=0.0)


def _lap1d_neumann(X: torch.Tensor) -> torch.Tensor:
    """1‑D Laplacian with Neumann (replicate) boundaries."""
    Xp = F.pad(X, (1, 1), mode="replicate")
    return Xp[:, :-2] - 2 * Xp[:, 1:-1] + Xp[:, 2:]


def _relax_ledger_once(
    X: torch.Tensor, *, lam: float, D: float, phi_inf: float, dt: float
) -> torch.Tensor:
    """One explicit Euler step of ∂tΦ = −λ(Φ−Φ∞) + D∇²Φ."""
    if lam <= 0 and D <= 0:
        return X
    lap = _lap1d_neumann(X)
    return X + dt * (-lam * (X - phi_inf) + D * lap)


def _apply_relaxation_effects(
    logits: torch.Tensor,
    X: torch.Tensor,
    relax: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict[str, float]]:
    """
    Optionally relax the ledger and compute fold counter ℱ.
    Returns: (logits, X_relaxed, F_or_None, meta)
    """
    eta = float(relax.get("eta", 0.0))
    w = float(relax.get("eta_path_weight", 0.0))
    rho = float(relax.get("rho", 0.0))  # folds→temperature coupling
    lam = float(relax.get("lambda", 0.0))
    D = float(relax.get("D", 0.0))
    phi_inf = float(relax.get("phi_inf", 0.0))
    steps = max(1, int(relax.get("steps", 1)))
    dt = float(relax.get("dt", 1.0))

    # Smooth ledger (tiny Euler steps)
    Xr = X
    if lam > 0 or D > 0:
        for _ in range(steps):
            Xr = _relax_ledger_once(Xr, lam=lam, D=D, phi_inf=phi_inf, dt=dt)

    # Fold counter along sequence
    F = _fold_counter(Xr, eta=eta, eta_path_weight=w) if (eta > 0 or w > 0) else None
    meta = {"eta": eta, "eta_path_weight": w, "rho": rho, "lambda": lam, "D": D, "dt": dt}
    return logits, Xr, F, meta


# ============================================================
# 3) Inference entry point
# ============================================================

def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cpu")


def infer_loop(
    model,
    x: Optional[torch.Tensor] = None,
    *,
    strategy: str = "greedy",
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    relax: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,  # NEW: reproducible sampling when strategy='sample'
) -> Dict[str, Any]:
    """
    Run one inference pass and decode tokens.

    Args
    ----
    model : ElementFold model returning (logits, X).
    x     : optional input (B,T) int64; if None, sampled uniformly using model.{vocab,seq_len}.
    strategy : 'greedy' | 'sample'.
    temperature, top_k, top_p : sampling knobs (used when strategy='sample').
    relax : optional dict enabling fold counter and ledger diffusion.
            Keys: {'eta','eta_path_weight','rho','lambda','D','phi_inf','steps','dt'}
    seed  : optional RNG seed for reproducible sampling.

    Returns
    -------
    dict with:
      'tokens' : (B,T) int64
      'ledger' : (B,T) same dtype as model ledger (possibly relaxed)
      optionally:
      'folds'      : (B,T) float tensor
      'relax_meta' : dict with applied relaxation scalars
    """
    if model is None:
        raise ValueError("infer_loop: model is None")

    device = _model_device(model)

    # Prepare input tokens
    if x is None:
        vocab = getattr(model, "vocab", None)
        seq_len = getattr(model, "seq_len", None)
        if vocab is None or seq_len is None:
            raise AttributeError("model must expose .vocab and .seq_len when x is None")
        x = torch.randint(0, int(vocab), (1, int(seq_len)), device=device, dtype=torch.long)
    else:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.to(device=device, dtype=torch.long, non_blocking=True)

    # Optional per-call sampling generator
    gen: Optional[torch.Generator] = None
    if seed is not None:
        gen = torch.Generator(device="cpu")  # generator is host-backed; works for multinomial
        gen.manual_seed(int(seed))

    was_training = getattr(model, "training", False)
    model.eval()

    with torch.inference_mode():
        logits, X = model(x)  # logits:(B,T,V), X:(B,T)

        # Optional relaxation (ledger smoothing + fold counter)
        if isinstance(relax, dict) and len(relax) > 0:
            logits, Xr, F, meta = _apply_relaxation_effects(logits, X, relax)
        else:
            Xr, F, meta = X, None, None

        # Decode
        if strategy == "greedy":
            y = logits.argmax(dim=-1)
        elif strategy == "sample":
            T_eff: float | torch.Tensor = temperature
            if isinstance(relax, dict) and relax and (F is not None):
                rho = float(relax.get("rho", 0.0))
                if rho != 0.0:
                    # Position-wise temperature lift with folds
                    T_eff = torch.exp(F.to(device) * rho) * float(temperature)
            y = _sample_from_logits(
                logits,
                temperature=T_eff,
                top_k=top_k,
                top_p=top_p,
                generator=gen,
            )
        else:
            raise ValueError(f"Unknown strategy {strategy!r}; expected 'greedy' or 'sample'.")

    if was_training:
        model.train()

    out: Dict[str, Any] = {"tokens": y, "ledger": Xr}
    if isinstance(relax, dict) and relax:
        out["folds"] = F if F is not None else torch.zeros_like(Xr)
        out["relax_meta"] = meta
    return out
