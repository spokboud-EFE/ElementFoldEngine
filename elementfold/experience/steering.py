# ElementFold · experience/steering.py
# The SteeringController turns *intent text* into a compact control vector:
#   v ∈ ℝ⁸ = [β, γ, clamp, style₅]
# where:
#   • β (beta)    — gate exposure strength (how strongly FGN exposes novelty),
#   • γ (gamma)   — normalization damping (how hard FGN calms energy),
#   • clamp (⛔)  — gate cap (how deep negative gate values can go before we clip),
#   • style₅      — free style slots adapters can use (e.g., tone, tempo, sharpness, etc.).
#
# The controller is deliberately small and fast:
#   tokenizer → ids → embedding → mean‑pool → 2‑layer MLP → ℝ⁸ control.
# We keep it trainable (see steering_train.py), but also useful “as is.”

import torch, torch.nn as nn                              # ✴ tensors • modules
from ..tokenizer import SimpleTokenizer                   # ✴ tiny tokenizer (vocab≈256)


class SteeringController(nn.Module):                      # 🎚 intent → (β, γ, ⛔, style₅)
    def __init__(self, delta: float = 0.030908106561043047):
        """
        Args:
            delta: δ⋆ coherence unit (cached here so downstream consumers can read it if needed).
        """
        super().__init__()                                # ✴ standard Module init
        self.delta = float(delta)                         # δ⋆ cached as a plain float

        # — Embedding —
        # We keep a tiny vocabulary (256) in sync with SimpleTokenizer; each token maps to ℝ⁶⁴.
        self.emb = nn.Embedding(256, 64)                  # E: vocab256 → ℝ⁶⁴

        # — Head (MLP) —
        # A small 2‑layer perceptron to turn the pooled embedding into ℝ⁸ (β, γ, ⛔, style₅).
        self.fc = nn.Sequential(                          # Π: ℝ⁶⁴ → ℝ⁸
            nn.Linear(64, 64),                            # affine → ℝ⁶⁴
            nn.ReLU(),                                    # nonlinearity (stable, simple)
            nn.Linear(64, 8),                             # affine → ℝ⁸
        )

        # We do not fix output ranges here; instead we offer a helper (to_params)
        # that maps raw outputs into meaningful ranges (β∈[0.5,2.0], γ∈[0,0.9], ⛔∈[1,10]).

    # ————————————————————————————————————————————————
    # Core forward (kept trainable for fine‑tuning)
    # ————————————————————————————————————————————————
    def forward(self, s: str) -> torch.Tensor:
        """
        Turn a text prompt into a raw control vector v ∈ ℝ⁸.

        Steps:
          1) tokenize the prompt (list[int]),
          2) embed tokens (1,L,64),
          3) mean‑pool across L → (1,64),
          4) MLP → (1,8),
          5) squeeze batch → (8,).

        Returns:
            torch.Tensor of shape (8,), dtype matches module parameters (float32 by default).
        """
        tok = SimpleTokenizer()                           # ✴ instantiate tokenizer
        ids = tok.encode(s)                               # ids: list[int], may be empty for empty input
        if len(ids) == 0:                                 # guard: ensure at least one token
            ids = [0]                                     # use a neutral token id 0

        # Build a tensor on the same device as our parameters to avoid device mismatches.
        dev = self.emb.weight.device                      # 🖥 where the module lives (cpu/cuda)
        x = torch.tensor(ids, dtype=torch.long, device=dev).unsqueeze(0)  # (1,L) batchify ids

        e = self.emb(x).mean(dim=1)                       # ⟲ pooled embedding (1,64) via mean over sequence length
        v = self.fc(e).squeeze(0)                         # ℝ⁸ = [β̂, γ̂, ⛔̂, style₅] (raw, unconstrained)
        return v                                          # ✴ raw controls (let caller map to ranges)

    # ————————————————————————————————————————————————
    # Helpers: map raw vector → meaningful ranges
    # ————————————————————————————————————————————————
    @staticmethod
    def to_params(v: torch.Tensor) -> dict:
        """
        Convert a raw ℝ⁸ vector (as returned by forward) into interpretable parameters
        with ranges aligned to the Supervisor’s defaults:

            β   ∈ [0.5, 2.0]
            γ   ∈ [0.0, 0.9]
            ⛔  ∈ [1.0, 10.0]
            style ∈ ℝ⁵  (left unconstrained; adapters interpret it)

        Returns:
            {'beta': float, 'gamma': float, 'clamp': float, 'style': torch.Tensor(5,)}
        """
        v = v.to(torch.float32)                           # ensure stable float math
        beta  = (v[0].sigmoid().item() + 0.5)            # map (−∞,∞) → (0,1) → (0.5,1.5) then +0.5 → (0.5,2.0)
        gamma = (v[1].sigmoid().item() * 0.9)            # (0,1) scaled into [0,0.9]
        clamp = (v[2].sigmoid().item() * 9.0 + 1.0)      # (0,1) → [1,10]
        style = v[3:8].detach()                           # pass style₅ as a small free vector for adapters
        return {"beta": beta, "gamma": gamma, "clamp": clamp, "style": style}

    # ————————————————————————————————————————————————
    # Convenience factories (untrained vs. checkpoint)
    # ————————————————————————————————————————————————
    @classmethod
    def load_default(cls, delta: float = 0.030908106561043047) -> "SteeringController":
        """
        Factory: create a fresh, untrained controller.
        This is useful for prototyping; training lives in steering_train.py.
        """
        return cls(delta)                                  # ≡ fresh controller (random weights)

    @classmethod
    def load(cls, path: str, delta: float = 0.030908106561043047) -> "SteeringController":
        """
        Factory: load weights from a state_dict checkpoint at `path`.
        The controller is returned in eval mode.
        """
        m = cls(delta)                                     # ✴ construct
        sd = torch.load(path, map_location="cpu")          # 🧱 read state dict (portable)
        m.load_state_dict(sd)                              # ⟲ load weights
        m.eval()                                           # ≡ evaluation mode (safer defaults)
        return m                                           # ✴ ready controller

    # ————————————————————————————————————————————————
    # Optional: apply controls to a model directly
    # ————————————————————————————————————————————————
    def apply_to_model(self, model, s: str | None = None, v: torch.Tensor | None = None) -> dict:
        """
        Convenience: produce controls (from a prompt `s` or raw vector `v`) and push them
        into any model that implements `.apply_control(beta=?, gamma=?, clamp=?)`.
        Returns the parameter dict actually applied.

        Usage:
            ctrl = SteeringController.load_default()
            applied = ctrl.apply_to_model(model, s="calm, softer, lower gain")
        """
        if v is None:
            if s is None:
                raise ValueError("either `s` (prompt) or `v` (raw ℝ⁸ vector) must be provided")
            v = self.forward(s)                             # ↦ raw ℝ⁸ from text
        params = self.to_params(v)                          # ↦ map into meaningful ranges
        if hasattr(model, "apply_control"):                 # only apply if the model supports it
            model.apply_control(beta=params["beta"], gamma=params["gamma"], clamp=params["clamp"])
        return params                                       # useful for logging/UX
