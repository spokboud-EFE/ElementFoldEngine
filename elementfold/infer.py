# ElementFold · infer.py
# ──────────────────────────────────────────────────────────────────────────────
# Inference is the calm twin of training: we take a model and (optionally) a token
# batch, run a forward pass, and turn logits into tokens. The ledger X comes along
# for telemetry. This file keeps things simple but robust:
#   • Auto‑select device + eval/inference mode.
#   • Greedy decoding by default (argmax).
#   • Optional temperature / top‑k / top‑p sampling for stochastic outputs.
#
# Notation (unicode used sparingly for clarity):
#   • β — “exposure” (lives inside the model; not touched here),
#   • γ — “damping”   (ditto),
#   • ⛔ — clamp       (ditto).
#
# We only *decode* here (logits → tokens); controlling β/γ/⛔ happens elsewhere.
#
# New (optional) in this revision — Relaxation clock (diffusion‑decay):
#   Pass a dict `relax={...}` to enable a tiny, safe “fold counter” ℱ and one
#   explicit smoothing step on the ledger (disabled by default).
#
#   Keys (all optional; conservative defaults):
#     relax = {
#       "eta": 0.0,              # base share‑rate per token step (folds per step)
#       "eta_path_weight": 0.0,  # extra share from path activity |ΔX| (0..1ish)
#       "rho": 0.0,              # how much ℱ increases sampling temperature: T_eff = T * exp(rho * ℱ)
#       "lambda": 0.0,           # letting‑go rate for a single explicit smoothing step on X
#       "D": 0.0,                # spatial diffusion strength along the sequence dim (Neumann boundaries)
#       "phi_inf": 0.0,          # calm baseline for letting‑go (Φ_∞)
#       "steps": 1,              # how many explicit Euler steps for the X‑smoothing (usually 1)
#       "dt": 1.0,               # step size; stability suggests D*dt ≤ 0.5 in 1‑D
#     }
#
#   Intuition:
#     • ℱ accumulates like “a little bit, many times” across token positions:
#         dℱ ≈ η · (1 + w·|ΔX|),  then  ℱ_t = Σ dℱ
#     • Larger ℱ gently raises sampling temperature (so distributions soften with distance).
#     • Optionally, X takes one tiny diffusion/letting‑go step:  ∂tΦ = −λ(Φ−Φ_∞) + D∇²Φ
#       This doesn’t affect tokens unless you *look* at the ledger, but it makes the
#       ledger’s “calming” visible in diagnostics.
#
from __future__ import annotations
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn.functional as F


# ————————————————————————————————————————————————————————————————
# Sampling utilities
# ————————————————————————————————————————————————————————————————

def _apply_topk(logits: torch.Tensor, k: Optional[int]) -> torch.Tensor:
    """
    Keep the top‑k logits per position, set the rest to −∞ (so softmax drops them).
    Shape: logits ∈ ℝ^{B×T×V}  →  returns a new tensor with same shape.
    """
    if k is None or k <= 0:
        return logits
    V = logits.size(-1)
    k = int(k)
    if k >= V:
        return logits  # nothing to do
    topk_vals, topk_idx = torch.topk(logits, k, dim=-1)
    pruned = torch.full_like(logits, float("-inf"))
    return pruned.scatter(-1, topk_idx, topk_vals)


def _apply_topp(logits: torch.Tensor, p: Optional[float]) -> torch.Tensor:
    """
    Nucleus (top‑p) filtering: keep the smallest prefix of tokens whose
    probability mass ≥ p at each (B,T) position; drop the rest (−∞).
    """
    if p is None or not (0.0 < float(p) < 1.0):
        return logits

    # Sort by logit for stable nucleus selection.
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)  # (B,T,V)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumprobs = probs.cumsum(dim=-1)

    # Mask everything AFTER the cutoff. Ensure the best token always survives.
    cutoff = cumprobs > float(p)
    cutoff[..., 0] = False

    kept = sorted_logits.masked_fill(cutoff, float("-inf"))
    # Scatter back to original token order.
    out = torch.full_like(logits, float("-inf"))
    return out.scatter(-1, sorted_idx, kept)


def _divide_by_temperature(logits: torch.Tensor, temperature: float | torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Support scalar or per‑position temperature:
      • float            → global T
      • tensor (B,T)     → per position
      • tensor (B,T,1)   → per position (already broadcastable)
    """
    if torch.is_tensor(temperature):
        T = temperature.to(dtype=logits.dtype, device=logits.device).clamp_min(eps)
        if T.dim() == 2:   # (B,T)
            T = T.unsqueeze(-1)  # (B,T,1)
        return logits / T
    else:
        return logits / max(float(temperature), eps)


def _sample_from_logits(
    logits: torch.Tensor,                   # (B,T,V) unnormalized scores
    temperature: float | torch.Tensor = 1.0,# float or (B,T)[,(B,T,1)]
    top_k: Optional[int] = None,            # keep K most likely tokens per position
    top_p: Optional[float] = None,          # nucleus sampling: keep minimal set with cumprob ≥ p
) -> torch.Tensor:
    """
    Turn logits into token ids using temperature + optional top‑k/top‑p filters.
    We operate position‑wise; the model predicts all positions in one pass.
    """
    # 0) Deterministic fallback: temperature ≤ 0 behaves like greedy (common UI edge case).
    if (not torch.is_tensor(temperature)) and (float(temperature) <= 0.0):
        return logits.argmax(dim=-1)

    # 1) Temperature scaling (supports scalar or (B,T)[,(B,T,1)]).
    logits = _divide_by_temperature(logits, temperature)

    # 2) Optional pruning. Order: top‑k then top‑p (both standard).
    logits = _apply_topk(logits, top_k)
    logits = _apply_topp(logits, top_p)

    # 3) Sample from categorical distribution at each (B,T) position.
    probs = torch.softmax(logits, dim=-1)                      # (B,T,V)
    B, T, V = probs.shape
    flat = probs.reshape(B * T, V)                             # (B·T, V)

    # Guard against numerical edge cases:
    nan_rows = torch.isnan(flat).any(dim=1)
    if nan_rows.any():
        flat[nan_rows] = 1.0 / V

    idx = torch.multinomial(flat, num_samples=1)               # (B·T,1)
    return idx.view(B, T)                                      # (B,T) int64


# ————————————————————————————————————————————————————————————————
# Relaxation helpers (optional)
# ————————————————————————————————————————————————————————————————

def _fold_counter(X: torch.Tensor, *, eta: float, eta_path_weight: float) -> torch.Tensor:
    """
    Compute a simple cumulative fold counter ℱ along the sequence:
        dℱ_t = η · (1 + w · |ΔX_t|),   ℱ_t = Σ_{τ≤t} dℱ_τ
    where |ΔX_t| ≈ |X_t − X_{t−1}| and w = eta_path_weight ≥ 0.
    Returns ℱ with shape (B,T), nonnegative and monotone in t.
    """
    if eta <= 0.0 and eta_path_weight <= 0.0:
        return torch.zeros_like(X)

    dX = torch.zeros_like(X)
    dX[:, 1:] = (X[:, 1:] - X[:, :-1]).abs()
    dF = float(eta) * (1.0 + float(eta_path_weight) * dX)
    F = torch.cumsum(dF, dim=1)
    # keep it finite and gentle
    return F.clamp_min(0.0)


def _lap1d_neumann(X: torch.Tensor) -> torch.Tensor:
    """
    1‑D discrete Laplacian along dim=1 with Neumann boundaries (replicate ends).
    """
    Xp = F.pad(X, (1, 1), mode="replicate")   # pad (left,right)
    return Xp[:, :-2] - 2.0 * Xp[:, 1:-1] + Xp[:, 2:]


def _relax_ledger_once(X: torch.Tensor, *, lam: float, D: float, phi_inf: float, dt: float) -> torch.Tensor:
    """
    Single explicit Euler step of:  ∂tΦ = −λ(Φ − Φ_∞) + D ∇²Φ   along the sequence.
    For stability in 1‑D with replicate boundaries, use D*dt ≤ 0.5 (we do not enforce,
    but defaults are tiny).
    """
    if (lam <= 0.0) and (D <= 0.0):
        return X
    lap = _lap1d_neumann(X)
    return X + float(dt) * ( - float(lam) * (X - float(phi_inf)) + float(D) * lap )


def _apply_relaxation_effects(
    logits: torch.Tensor,  # (B,T,V)
    X: torch.Tensor,       # (B,T)
    relax: Dict[str, Any]  # user knobs (already a dict)
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict[str, float]]:
    """
    Compute:
      • optional one‑step relaxed ledger X_relaxed,
      • fold counter ℱ,
      • per‑position temperature multiplier via  T_eff = T_base * exp(ρ · ℱ).
    Returns: (logits, X_relaxed, F, meta)
    """
    # Read knobs with safe defaults
    eta = float(relax.get("eta", 0.0))
    w   = float(relax.get("eta_path_weight", 0.0))
    rho = float(relax.get("rho", 0.0))

    lam = float(relax.get("lambda", 0.0))
    D   = float(relax.get("D", 0.0))
    phi_inf = float(relax.get("phi_inf", 0.0))
    steps = int(relax.get("steps", 1))
    dt    = float(relax.get("dt", 1.0))

    # 1) Optional ledger smoothing (explicit Euler; typically 1 tiny step).
    Xr = X
    if (lam > 0.0) or (D > 0.0):
        # Conservative stability hint (not enforced): D*dt ≤ 0.5
        for _ in range(max(1, steps)):
            Xr = _relax_ledger_once(Xr, lam=lam, D=D, phi_inf=phi_inf, dt=dt)

    # 2) Fold counter along the sequence (used for temperature lift).
    F = _fold_counter(Xr, eta=eta, eta_path_weight=w) if (eta > 0.0 or w > 0.0) else None

    # 3) Package a tiny meta summary (handy for logs)
    meta = {"eta": eta, "eta_path_weight": w, "rho": rho, "lambda": lam, "D": D, "dt": dt}

    return logits, Xr, F, meta


# ————————————————————————————————————————————————————————————————
# Main inference entry
# ————————————————————————————————————————————————————————————————

def _model_device(model) -> torch.device:
    """Best‑effort device discovery even for exotic models."""
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cpu")


def infer_loop(
    model,
    x: Optional[torch.Tensor] = None,   # optional input tokens: (B,T) int64
    *,
    strategy: str = "greedy",           # 'greedy' or 'sample'
    temperature: float = 1.0,           # used when strategy='sample'
    top_k: Optional[int] = None,        # used when strategy='sample'
    top_p: Optional[float] = None,      # used when strategy='sample'
    relax: Optional[Dict[str, Any]] = None,  # optional diffusion‑decay knobs (see header)
) -> Dict[str, Any]:
    """
    Run a forward pass and decode tokens with the chosen strategy.

    Inputs
    ------
    model:
        ElementFold model whose forward returns (logits, X), where:
          • logits ∈ ℝ^{B×T×V} (unnormalized scores over vocab),
          • X      ∈ ℝ^{B×T}   (ledger: latent coordinate along rungs).
    x:
        Optional input tokens (B,T), dtype long. If None, we synthesize a random
        (1,T) batch using model.vocab and model.seq_len (handy for demos).
    strategy:
        'greedy' (argmax; deterministic) or 'sample' (stochastic).
    temperature, top_k, top_p:
        Sampling knobs when strategy='sample' (ignored for greedy).
    relax:
        Optional dict to enable the “relaxation clock”:
          • computes a fold counter ℱ from the ledger,
          • optionally diffuses the ledger once,
          • for sampling, raises the temperature per position:  T_eff = T · exp(ρ·ℱ).

    Returns
    -------
    dict with:
      • 'tokens' : (B,T) int64 — decoded tokens,
      • 'ledger' : (B,T) float — (possibly relaxed) ledger X,
      • 'folds'  : (B,T) float — cumulative folds ℱ (present only if relax supplied),
      • 'relax_meta' : dict    — tiny echo of knobs used (present only if relax supplied).
    """
    if model is None:
        raise ValueError("infer_loop: model is None")

    device = _model_device(model)

    # Prepare input: ensure (B,T) on the correct device.
    if x is None:
        # Rely on model attributes (ElementFold Model exposes .vocab and .seq_len).
        vocab = getattr(model, "vocab", None)
        seq_len = getattr(model, "seq_len", None)
        if vocab is None or seq_len is None:
            raise AttributeError("model must expose .vocab and .seq_len when x is None")
        x = torch.randint(0, int(vocab), (1, int(seq_len)), device=device)
    else:
        if x.dim() == 1:
            x = x.unsqueeze(0)  # → (1,T)
        x = x.to(device)

    was_training = getattr(model, "training", False)
    model.eval()
    with torch.inference_mode():
        logits, X = model(x)  # (B,T,V), (B,T)

        # Optional relaxation effects (purely post‑forward; model untouched).
        if isinstance(relax, dict) and relax:
            logits, Xr, F, meta = _apply_relaxation_effects(logits, X, relax)
        else:
            Xr, F, meta = X, None, None

        if strategy == "greedy":
            y = logits.argmax(dim=-1)  # (B,T)
        elif strategy == "sample":
            # If we have folds and a positive rho, lift temperature per position:
            T_eff = temperature
            if isinstance(relax, dict) and relax and (F is not None):
                rho = float(relax.get("rho", 0.0))
                if rho != 0.0:
                    # T_eff = T_base * exp(rho * F)
                    T_eff = torch.exp(F.to(device) * float(rho)) * float(temperature)
            y = _sample_from_logits(
                logits,
                temperature=T_eff,
                top_k=(int(top_k) if top_k is not None else None),
                top_p=(float(top_p) if top_p is not None else None),
            )
        else:
            raise ValueError(f"unknown strategy: {strategy!r}; use 'greedy' or 'sample'")

    if was_training:
        model.train()

    out: Dict[str, Any] = {"tokens": y, "ledger": Xr}
    if isinstance(relax, dict) and relax:
        if F is None:
            # Provide an all‑zero ℱ for uniformity if relax was passed but eta=0
            F = torch.zeros_like(Xr)
        out["folds"] = F
        out["relax_meta"] = meta
    return out
