# ElementFold · experience/steering_train.py
# Train the tiny SteeringController on simple synthetic pairs:
#     prompt  →  target v ∈ ℝ⁸ = [β, γ, clamp, style₅]
#
# Design goals (plain words):
#   • Tiny & dependency‑free — runs anywhere PyTorch does.
#   • Teaches the *same mapping* used at runtime: raw ℝ⁸ → (β, γ, ⛔, style₅).
#     (We optimize the *mapped* parameters so Studio/Engine behavior matches training.)
#   • Gentle defaults; a single file you can read top‑to‑bottom.
#
# Usage:
#   from elementfold.experience.steering_train import fit_steering
#   ctrl = fit_steering(steps=800, save_path="runs/steering/ctrl.pt")
#   # later: SteeringController.load("runs/steering/ctrl.pt")

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from .steering import SteeringController  # 🎚 intent → raw ℝ⁸ (we map it to β,γ,⛔,style₅ below)


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic supervision (tiny, fast, explanatory)
# ──────────────────────────────────────────────────────────────────────────────

def make_synth(n: int = 512, seed: int | None = None) -> Tuple[List[str], torch.Tensor]:
    """
    Build a small synthetic dataset of (prompt, target) pairs.

    Prompts are style tags; targets are noisy vectors:
        v = [β, γ, clamp, style₅]
    where β∈[0.5,1.5], γ∈[0,0.9], clamp∈[1,10] roughly match runtime ranges.

    Returns:
      prompts: list[str] length n
      targets: float32 tensor (n, 8)
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    # Simple prototypes near the controller’s operating ranges.
    # (Feel free to add your own tags later.)
    prototypes = [
        ("calm",  0.60, 0.20, 3.0),
        ("sharp", 1.20, 0.40, 6.0),
        ("bold",  1.40, 0.60, 8.0),
        ("soft",  0.80, 0.30, 4.0),
    ]

    prompts: List[str] = []
    targets: List[torch.Tensor] = []

    for _ in range(int(n)):
        tag, b, g, c = random.choice(prototypes)
        beta  = b + 0.10 * (random.random() - 0.5)   # small jitter
        gamma = g + 0.05 * (random.random() - 0.5)
        clamp = c + 1.00 * (random.random() - 0.5)
        style = torch.randn(5) * 0.1                 # lightweight free style vector
        v = torch.tensor([beta, gamma, clamp, *style.tolist()], dtype=torch.float32)  # ℝ⁸ target
        prompts.append(tag)
        targets.append(v)

    return prompts, torch.stack(targets, dim=0)      # (list[str], (n,8))


# ──────────────────────────────────────────────────────────────────────────────
# Internal: differentiable mapping raw ℝ⁸ → (β, γ, ⛔, style₅)
# Mirrors SteeringController.to_params — but keeps gradients for training.
# ──────────────────────────────────────────────────────────────────────────────

def _map_raw_to_params(v: torch.Tensor) -> torch.Tensor:
    """
    Map raw controller outputs v (…,8) to the runtime parameter ranges
    without detaching, so gradients flow during training:

        β     = σ(v₀) + 0.5          ∈ (0.5, 1.5)   [matches current runtime mapping]
        γ     = 0.9 · σ(v₁)          ∈ (0.0, 0.9)
        clamp = 1 + 9 · σ(v₂)         ∈ (1.0, 10.0)
        style = v₃:₈                  ∈ ℝ⁵ (free)

    Returns:
        Tensor of shape (…,8) with the mapped parameters.
    """
    v = v.to(torch.float32)
    beta  = torch.sigmoid(v[..., 0]) + 0.5                             # (…,)
    gamma = torch.sigmoid(v[..., 1]) * 0.9                              # (…,)
    clamp = torch.sigmoid(v[..., 2]) * 9.0 + 1.0                        # (…,)
    style = v[..., 3:8]                                                 # (…,5)
    head = torch.stack([beta, gamma, clamp], dim=-1)                    # (…,3)
    return torch.cat([head, style], dim=-1)                             # (…,8)


# ──────────────────────────────────────────────────────────────────────────────
# Training loop (small, portable, with best‑val selection)
# ──────────────────────────────────────────────────────────────────────────────

def fit_steering(
    steps: int = 500,                 # Total optimization steps
    lr: float = 1e-3,                 # Adam learning rate
    save_path: str | None = None,     # Optional checkpoint path (file or directory)
    delta: float = 0.030908106561043047,  # δ⋆ (cached into the controller for UX)
    batch_size: int = 16,             # Mini‑batch size
    seed: int | None = 1234,          # RNG seed for reproducibility
    val_frac: float = 0.1,            # Validation split fraction
    device: str | None = None,        # 'cuda' | 'cpu' | None (auto)
    print_every: int | None = None,   # e.g., 50 for progress prints; None = silent
) -> SteeringController:
    """
    Train a SteeringController on synthetic pairs and optionally save weights.

    Returns:
        Trained SteeringController in eval() mode (ready for SteeringController.load()).
    """
    # 0) Device & RNG
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    # 1) Data: train/val split
    prompts, targets = make_synth(n=512, seed=seed)
    n = len(prompts)
    n_val = max(1, int(val_frac * n))
    idx = list(range(n))
    random.shuffle(idx)
    val_idx = set(idx[:n_val])
    train_data = [(prompts[i], targets[i]) for i in range(n) if i not in val_idx]
    val_data   = [(prompts[i], targets[i]) for i in range(n) if i in val_idx]

    # 2) Model + optimizer
    m = SteeringController(delta).to(device)
    m.train()
    opt = optim.Adam(m.parameters(), lr=lr)

    # Small helpers
    def _sample_batch(data, k: int):
        items = [data[random.randrange(0, len(data))] for _ in range(k)]
        prompts_b = [s for (s, _) in items]
        targets_b = torch.stack([t for (_, t) in items]).to(device)
        return prompts_b, targets_b

    @torch.no_grad()
    def _eval_mse(data) -> float:
        m.eval()
        total = 0.0
        count = 0
        # Chunk the evaluation to avoid large host lists
        chunk = max(batch_size, 64)
        for i in range(0, len(data), chunk):
            prompts_c = [s for (s, _) in data[i:i+chunk]]
            targets_c = torch.stack([t for (_, t) in data[i:i+chunk]]).to(device)
            raw = torch.stack([m(s) for s in prompts_c])            # (C,8) raw
            pred = _map_raw_to_params(raw)                          # (C,8) mapped
            loss = (pred - targets_c).pow(2).mean().item()
            total += loss * len(prompts_c)
            count += len(prompts_c)
        m.train()
        return total / max(1, count)

    # 3) Train with best‑val checkpointing
    best_val = math.inf
    best_state = None

    for step in range(int(steps)):
        prompts_b, target = _sample_batch(train_data, batch_size)
        raw = torch.stack([m(s) for s in prompts_b])                # (B,8) raw controller output
        pred = _map_raw_to_params(raw)                              # (B,8) map to (β,γ,⛔,style₅)
        loss = (pred - target).pow(2).mean()                        # MSE in the *mapped* parameter space

        opt.zero_grad(set_to_none=True)
        loss.backward()
        # Small, safe clip to avoid occasional spikes on fresh seeds
        torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
        opt.step()

        # Periodic validation / best snapshot
        is_last = (step == steps - 1)
        if (print_every is not None and (step + 1) % print_every == 0) or is_last:
            val_mse = _eval_mse(val_data)
            if val_mse < best_val:
                best_val = val_mse
                best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
            if print_every is not None:
                print(f"[steering] step {step+1:4d}/{steps}  train_mse={loss.item():.4f}  val_mse={val_mse:.4f}")

    # 4) Restore best weights and save (optional)
    if best_state is not None:
        m.load_state_dict(best_state)
    m.eval()

    if save_path is not None:
        path = Path(save_path)
        if path.is_dir() or not path.suffix:         # directory or extensionless → write checkpoint.pt inside it
            path.mkdir(parents=True, exist_ok=True)
            path = path / "checkpoint.pt"
        torch.save(m.state_dict(), path)
        print(f"✓ SteeringController saved to {path}")
    else:
        print("✓ SteeringController training done (no save path provided)")

    return m
