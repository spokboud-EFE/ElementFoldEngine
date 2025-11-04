# ElementFold · experience/steering_train.py
# This file trains the SteeringController on simple synthetic pairs:
#   prompt  →  target vector v ∈ ℝ⁸ = [β, γ, clamp, style₅]
# It’s deliberately small, dependency‑free, and narrated so a non‑expert can follow.
# Use it:
#   from elementfold.experience.steering_train import fit_steering
#   ctrl = fit_steering(steps=1000, save_path="ctrl.pt")
#   # then: SteeringController.load("ctrl.pt")

import math
import random
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim      # tensors • modules • optimizers
from .steering import SteeringController                # 🎚 intent → control vector


# ————————————————————————————————————————————————
# Synthetic supervision (tiny, fast, good enough to demo)
# ————————————————————————————————————————————————

def make_synth(n: int = 512, seed: int | None = None) -> tuple[list[str], torch.Tensor]:
    """
    Build a tiny synthetic dataset of (prompt, target) pairs.

    Each prompt is one of a few style tags; the target is a noisy vector:
        v = [β, γ, clamp, style₅]  where style₅ ~ N(0, 0.1²)

    Args:
        n:     number of samples to generate
        seed:  optional RNG seed for reproducibility

    Returns:
        prompts: list[str] length n
        targets: torch.Tensor (n,8) with dtype float32
    """
    if seed is not None:                               # Reproducibility on demand
        random.seed(seed)
        torch.manual_seed(seed)

    # Simple style prototypes: (tag, β, γ, ⛔)
    keys = [
        ("calm",  0.6, 0.2, 3.0),
        ("sharp", 1.2, 0.4, 6.0),
        ("bold",  1.4, 0.6, 8.0),
        ("soft",  0.8, 0.3, 4.0),
    ]

    prompts: list[str] = []
    targets: list[torch.Tensor] = []

    for _ in range(int(n)):                             # Sample n points
        tag, b, g, c = random.choice(keys)             # Choose a base prototype
        beta  = b + 0.1  * (random.random() - 0.5)     # Add small jitter per dimension
        gamma = g + 0.05 * (random.random() - 0.5)
        clamp = c + 1.0  * (random.random() - 0.5)
        style = torch.randn(5) * 0.1                   # Style slots get light Gaussian noise
        v = torch.tensor([beta, gamma, clamp, *style.tolist()], dtype=torch.float32)  # Build ℝ⁸
        prompts.append(tag)
        targets.append(v)

    return prompts, torch.stack(targets, dim=0)         # (list[str], (n,8))


# ————————————————————————————————————————————————
# Training loop (tiny, vectorized, portable)
# ————————————————————————————————————————————————

def fit_steering(
    steps: int = 500,               # Total optimization steps
    lr: float = 1e-3,               # Adam learning rate
    save_path: str | None = None,   # Optional checkpoint path
    delta: float = 0.030908106561043047,  # δ⋆ coherence unit for controller context
    batch_size: int = 16,           # Mini‑batch size
    seed: int | None = 1234,        # RNG seed for reproducibility
    val_frac: float = 0.1,          # Fraction of data used for validation
    device: str | None = None,      # 'cuda'|'cpu' (auto if None)
    print_every: int | None = None, # e.g., 50 for progress prints; None = silent
) -> SteeringController:
    """
    Train a SteeringController on synthetic pairs and optionally save weights.

    Returns:
        Trained SteeringController in eval() mode.
    """
    # 0) Device selection and RNG
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    # 1) Data: split into train/val
    prompts, targets = make_synth(n=512, seed=seed)
    n = len(prompts)
    n_val = max(1, int(val_frac * n))
    idx_all = list(range(n))
    random.shuffle(idx_all)
    idx_val = set(idx_all[:n_val])
    train_data = [(prompts[i], targets[i]) for i in range(n) if i not in idx_val]
    val_data   = [(prompts[i], targets[i]) for i in range(n) if i in idx_val]

    # 2) Model + optimizer
    m = SteeringController(delta).to(device)
    m.train()
    opt = optim.Adam(m.parameters(), lr=lr)

    # Small helpers: batched sampling and evaluation
    def sample_batch(data, k):
        # Randomly draw k items (with replacement) from the given list of (prompt, target).
        out = [data[random.randrange(0, len(data))] for _ in range(k)]
        batch_prompts = [s for (s, _) in out]
        batch_targets = torch.stack([t for (_, t) in out]).to(device)
        return batch_prompts, batch_targets

    @torch.no_grad()
    def eval_mse(data) -> float:
        # Evaluate mean‑squared error over the given dataset.
        m.eval()
        total = 0.0
        count = 0
        # Process in manageable chunks without building a full token batcher
        chunk = max(1, 64 // batch_size) * batch_size  # simple chunk size ≈64
        for i in range(0, len(data), chunk):
            chunk_prompts = [s for (s, _) in data[i:i+chunk]]
            chunk_targets = torch.stack([t for (_, t) in data[i:i+chunk]]).to(device)
            # Vectorize via Python list (controller.forward handles tokenization internally).
            out = torch.stack([m(s) for s in chunk_prompts])      # (C,8)
            loss = (out - chunk_targets).pow(2).mean().item()
            total += loss * len(chunk_prompts)
            count += len(chunk_prompts)
        m.train()
        return total / max(1, count)

    # 3) Training loop with best‑val checkpoint
    best_val = math.inf
    best_state = None

    for step in range(int(steps)):
        batch_prompts, target = sample_batch(train_data, batch_size)   # Prompts list + (B,8) targets
        out = torch.stack([m(s) for s in batch_prompts])               # Forward controller on each prompt → (B,8)
        loss = (out - target).pow(2).mean()                            # MSE regression loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # Periodic validation/checkpoint
        if (print_every is not None and (step + 1) % print_every == 0) or step == steps - 1:
            val_mse = eval_mse(val_data)
            if val_mse < best_val:
                best_val = val_mse
                best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
            # Optional progress readout
            if print_every is not None:
                print(f"[steering] step {step+1:4d}/{steps}  train_mse={loss.item():.4f}  val_mse={val_mse:.4f}")

    # 4) Restore best weights (if we improved) and save optionally
    if best_state is not None:
        m.load_state_dict(best_state)
    m.eval()
    if save_path is not None:
        path = Path(save_path)
        # Treat paths without an extension as directories by default
        if path.is_dir() or not path.suffix:
            path.mkdir(parents=True, exist_ok=True)
            path = path / "checkpoint.pt"
        torch.save(m.state_dict(), path)
        print(f"✓ SteeringController saved to {path}")
    else:
        print("✓ SteeringController training done (no save path provided)")
    return m
