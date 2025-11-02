# ElementFold · train.py
# This is the end‑to‑end training loop. Read it like a story:
#   1) We build the model and its two “physics” companions:
#        • AlignHead   — keeps phases aligned on the δ⋆ circle without a temperature.
#        • Variational — keeps seats and blocks equally spaced (convex backbone).
#   2) We create an optimizer (AdamW with clean param groups) and a gentle LR schedule.
#   3) Each step, we fetch a batch (or synthesize tokens), run the model, compute three losses:
#        ℒ_task  (cross‑entropy on tokens),
#        ℒ_align (temperature‑free contrast on the circle),
#        E_ledger (convex spacing energy),
#      then combine them into one total loss.
#   4) We backprop, clip gradients (safety), and step the optimizer and scheduler.
#   5) We measure coherence (κ, p½, margins) and let the Supervisor nudge β, γ, and ⛔.
#   6) We push those controls into the model (apply_control) so the next step is more stable.

import torch                                     # Tensors, CUDA checks
from torch.nn.utils import clip_grad_norm_       # Safety clamp on ∥∇∥
from .model import Model                         # The coherent core (RotaryClick + FGN stack)
from .align import AlignHead                     # Temperature‑free contrastive alignment
from .variational import VariationalLedger       # Convex ledger spacing energy
from .telemetry import measure                   # κ, p½, margins, residual stats
from .optim import build_optimizer, make_scheduler, get_lr  # AdamW + warmup/cosine schedule
from .control import Supervisor                  # Feedback controller for β, γ, ⛔
from .data import DataLoaderBuilder              # Minimal data source (or synth)


def train_loop(
    device=None,                                 # Which accelerator? Auto‑select if None.
    steps=200,                                   # How many optimization steps to run.
    vocab=256, d=128, layers=4, heads=4,         # Model shape (heads kept for parity with attention configs).
    seq_len=128, fold='grid',                    # Sequence length and fold kind (grid FGN here).
    delta=0.03,                                  # The click δ⋆ (controls rotary angle and circular geometry).
    capacities=(2, 6, 10, 14),                   # Seat capacities per block (used by Variational + Align).
    batch=32, use_data=True,                     # Batch size and whether to use a DataLoader.
    lr=2e-4, wd=0.01,                            # Optimizer learning rate and weight decay.
    warmup_frac=0.1,                             # Fraction of steps used for warmup before cosine decay.
    clip_norm=1.0,                               # Gradient norm clamp for safety.
    tv_weight=0.0,                               # Total‑variation weight in the variational energy.
):
    """
    ⟲ ElementFold training loop — *why these numbers* (defaults chosen for stability & portability).

    Device
    -------
    device = None
        • Auto‑select 'cuda' if available else 'cpu'. Keeps notebooks and headless servers zero‑config.

    Iteration Budget
    ----------------
    steps = 200
        • Small but meaningful budget to reach “coherence visibly emerges” in smoke tests.
        • Scale up for real runs: 2–10k on a single GPU; keep warmup_frac fixed (see below).

    Model Shape
    -----------
    vocab = 256
        • Byte‑level vocabulary (0..255) matches text/image/audio tokenizations used across the repo.
    d = 128
        • Feature width. 128 is the sweet spot for fast iteration on CPU/GPU with FGN blocks.
        • Bigger d improves capacity but quadratically increases compute in linear layers.
    layers = 4
        • Depth of the Fold–Gate–Norm (FGN) stack. Four layers give room for refinement without instability.
    heads = 4
        • Kept for config parity with attention‑style models and future experiments (not used by FGN).

    Sequence Geometry
    -----------------
    seq_len = 128
        • Balanced context vs. memory footprint; pairs well with small batches on a single GPU.
    fold = 'grid'
        • Selects the time‑grid fold (depthwise conv). Other folds (e.g., graph) plug in at the same interface.

    Rotary Click
    ------------
    delta = 0.03  (δ⋆)
        • Determines θ⋆ = 2π·δ⋆, the per‑step rotation used by RotaryClick.
        • Small δ⋆ → slow phase advance across time (gentler; easier to lock); larger δ⋆ → faster rotation.
        • Typical range: 0.02–0.05. We use ~0.03 as a default that mixes well with capacities below.

    Ledger Capacities
    -----------------
    capacities = (2, 6, 10, 14)
        • Seat counts used by the VariationalLedger and Align head.
        • Mix of small composite numbers touches prime factors {2,3,5,7}:
            2  → even symmetry
            6  → 2×3
            10 → 2×5
            14 → 2×7
          This “harmonic palette” makes misalignment detectable across multiple modular resolutions.
        • If you only want tiny cycles, use (2,4,8); if you want broader structure, add (12,20).

    Batch & Data
    ------------
    batch = 32
        • Default mini‑batch that keeps gradient statistics stable on a single consumer GPU.
        • On CPU or tiny GPUs: 8–16. On large GPUs: 64–256 (watch memory).
    use_data = True
        • True → real DataLoader; False → synthetic random tokens (quick smoke test).

    Optimizer & Regularization
    --------------------------
    lr = 2e-4
        • AdamW step size that behaves well for d≈128, layers≈4 with warmup.
        • If you disable warmup, reduce to ~1e‑4.
    wd = 0.01
        • Weight decay on weight matrices only (bias/norm excluded). Encourages smoother minima.
    warmup_frac = 0.1
        • 10% of steps linearly ramp lr from 0 → 1×lr, then cosine decay thereafter.
        • Rationale: FGN gates can briefly over‑expose; warmup prevents jolting the optimizer early.
        • Keep between 0.05 and 0.2 for most runs.
    clip_norm = 1.0
        • Global gradient‑norm clamp (L2). A safety rail: a single spiky batch won’t explode momentum.
        • If you see chronic clipping (telemetry), lower lr or raise clip_norm slightly (e.g., 1.5).
    tv_weight = 0.0
        • Multiplier on the 1‑D total‑variation penalty inside the VariationalLedger.
        • Use 1e‑3…1e‑2 if you want a smoother ledger trajectory X; keep 0.0 for unconstrained learning.

    Notes on Loss Balancing (inside the loop)
    -----------------------------------------
    • Task loss: cross‑entropy on logits vs. tokens — anchors the model to the data.
    • Align loss: temperature‑free contrast on the circle — encourages δ⋆‑coherence.
    • Variational energy: equal‑spacing + block spacing (+ optional TV) — gives a convex “shape” prior.

      A practical starting blend is:
          loss = L_task
               + 1.0 * L_align
               + 0.1 * E_variational / (batch * seq_len)

      Where the last term is normalized by tokens so the weight is roughly scale‑free.

    Quick Recipes
    -------------
    CPU smoke test:
        steps=120, d=64, layers=2, batch=8, seq_len=96, lr=1e-4, warmup_frac=0.2
    Small GPU (e.g., T4):
        steps=2_000, d=128, layers=4, batch=32, seq_len=128, lr=2e-4, warmup_frac=0.1
    Larger GPU:
        steps=10_000, d=256, layers=6, batch=128, seq_len=256, lr=2e-4 (watch memory)

    Telemetry
    ---------
    • κ (kappa): |⟨e^{i·2πX/δ⋆}⟩| — higher means stronger phase concentration (more coherent).
    • p½: fraction near the half‑click boundary — if >5%, increase γ (damping) or tighten clamp.

    """

    # ——— 1) Pick device (CUDA if available) ————————————————————————————————
    if device is None:                           # If caller didn’t specify a device …
        device = 'cuda' if torch.cuda.is_available() else 'cpu'  # … pick CUDA when possible.

    # ——— 2) Build model and companions ————————————————————————————————
    model = Model(vocab=vocab, d=d, layers=layers, heads=heads, seq_len=seq_len, fold=fold, delta=delta).to(device)
    align = AlignHead(delta).to(device)          # Alignment head shares δ⋆ (temperature‑free geometry).
    var = VariationalLedger(delta, capacities, tv_weight=float(tv_weight)).to(device)  # Convex spacing energy.

    # ——— 3) Optimizer + gentle LR schedule ————————————————————————————————
    opt = build_optimizer(model, lr=lr, wd=wd)   # AdamW with decay/no‑decay param groups.
    warmup_steps = max(1, int(warmup_frac * steps))                  # A small ramp to avoid cold starts.
    scheduler = make_scheduler(opt, warmup_steps=warmup_steps, total_steps=steps, min_lr_scale=0.1)  # Cosine after warmup.

    # ——— 4) Supervisor (β exposure, γ damping, ⛔ clamp) ————————————————————
    sup = Supervisor()                           # Starts at β=1.0, γ=0.5, ⛔=5.0 by default.

    # ——— 5) Data stream (or synthetic tokens) ————————————————————————————
    if use_data:
        dl = DataLoaderBuilder(seq_len=seq_len, vocab=vocab, batch=batch).make()  # A tiny loader that yields (B,T) int64
        it = iter(dl)                          # Create an iterator we can rewind on exhaustion.

    # ——— 6) Optimization loop ————————————————————————————————————————————
    for step in range(steps):
        # 6.a) Fetch a batch (wrap the iterator cleanly)
        if use_data:
            try:
                x = next(it).to(device)         # (B,T) token ids on the right device
            except StopIteration:               # If we ran out of data, rewind
                it = iter(dl)
                x = next(it).to(device)
        else:
            x = torch.randint(0, vocab, (batch, seq_len), device=device)  # Synthetic tokens (useful for smoke tests)

        # 6.b) Forward pass: logits and ledger scalars X per time step
        logits, X = model(x)                    # logits: (B,T,V), X: (B,T)

        # 6.c) Losses: task (CE), alignment (temperature‑free NCE), variational (convex energy)
        loss_task = torch.nn.functional.cross_entropy(logits.view(-1, vocab), x.view(-1))  # Language modeling CE
        loss_align, pos, neg = align(X.mean(dim=1), torch.as_tensor(capacities, device=device))  # κ‑based contrast
        e = var.energy(X[:, :max(capacities)])  # Enforce equal seat spacing inside each block and δ⋆ across blocks

        # Combine with small weights so each term “speaks” but task remains primary
        loss = loss_task + 1.0 * loss_align + 0.1 * e / (batch * seq_len)

        # 6.d) Backward pass with safety rails (zero‑grad → backprop → clip → step)
        opt.zero_grad(set_to_none=True)        # Drop old gradient buffers (faster than filling zeros)
        loss.backward()                        # Compute ∇ for all trainable parameters
        grad_norm = float(clip_grad_norm_(model.parameters(), clip_norm))  # Clamp ∥∇∥₂ ≤ clip_norm; record for telemetry
        opt.step()                              # AdamW update (decoupled weight decay)
        scheduler.step()                        # LR schedule tick (warmup → cosine)

        # 6.e) Read coherence telemetry (κ, p½, margins) and update Supervisor
        tele = measure(X.mean(dim=1), delta, detail=False)     # Summarize coherence per batch
        tele["grad_norm"] = grad_norm                          # Add gradient norm for stability hints
        ctrl = sup.update(tele)                                # Adjust β, γ, ⛔ recommendations
        if hasattr(model, "apply_control"):                    # Push controls into the FGN blocks if supported
            model.apply_control(beta=ctrl["beta"], gamma=ctrl["gamma"], clamp=ctrl["clamp"])

        # (Optional) Light on‑step prints could be added by caller; we keep core loop quiet by default.

    return model  # Trained model ready for inference or saving
