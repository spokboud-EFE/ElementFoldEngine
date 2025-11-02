# ElementFold · infer.py
# Inference is the calm twin of training: we take a model and an optional token batch,
# run a forward pass, and turn logits into tokens. The ledger X comes along for telemetry.
# This file keeps things simple but robust:
#   • Auto‑select device and eval/inference mode.
#   • Greedy decoding by default (argmax).
#   • Optional temperature / top‑k / top‑p sampling if you want stochastic outputs.

import torch  # Tensors and inference context


def _sample_from_logits(
    logits: torch.Tensor,           # (B,T,V) unnormalized scores
    temperature: float = 1.0,       # >0; lower → sharper, higher → flatter
    top_k: int | None = None,       # keep K most likely tokens per position
    top_p: float | None = None,     # nucleus sampling: keep smallest set with cumprob ≥ p
) -> torch.Tensor:
    """
    Turn logits into token ids using temperature + (optional) top‑k/top‑p.
    We operate position‑wise; this model predicts all positions in one pass.
    """
    # 1) Optional temperature scaling (never divide by 0)
    if temperature <= 0:
        temperature = 1e-8  # guard against zero/negative values
    logits = logits / float(temperature)  # sharper or flatter distribution

    # 2) Optional top‑k filter: zero out everything except the top k per position
    if top_k is not None and top_k > 0:
        k = min(int(top_k), logits.size(-1))                     # clamp K to vocab size
        topk_vals, topk_idx = torch.topk(logits, k, dim=-1)      # top‑k per (B,T)
        mask = torch.full_like(logits, float("-inf"))            # start with −∞ everywhere
        logits = mask.scatter(-1, topk_idx, topk_vals)           # place top‑k values back

    # 3) Optional top‑p (nucleus): keep the smallest prefix reaching probability mass ≥ p
    if top_p is not None and 0.0 < float(top_p) < 1.0:
        # Sort tokens by logit descending to build a cumulative prob tail
        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)                      # convert to probs
        cumprobs = probs.cumsum(dim=-1)                                   # cumulative probability
        # Build a mask of tokens to drop (everything after the cutoff)
        cutoff = (cumprobs > float(top_p))                                # True where past nucleus
        # Ensure at least one token is kept at each position
        cutoff[..., 0] = False
        # Replace dropped token logits with −∞; then scatter back to original order
        sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)

    # 4) Sample from the (filtered) categorical distribution
    probs = torch.softmax(logits, dim=-1)                       # proper distribution over tokens
    # torch.multinomial needs 2‑D, so we flatten (B*T, V) and then unflatten back
    B, T, V = probs.shape
    flat = probs.reshape(B * T, V)
    idx = torch.multinomial(flat, num_samples=1)                # draw 1 token per position
    return idx.view(B, T)                                       # (B,T) token ids


def infer_loop(
    model,
    x: torch.Tensor | None = None,      # optional input tokens: (B,T) int64
    strategy: str = "greedy",           # 'greedy' or 'sample'
    temperature: float = 1.0,           # used when strategy='sample'
    top_k: int | None = None,           # used when strategy='sample'
    top_p: float | None = None,         # used when strategy='sample'
):
    """
    Run a forward pass and decode tokens with the chosen strategy.

    Returns:
        {
          'tokens': (B,T) int64 decoded tokens,
          'ledger': (B,T) float32 ledger coordinates X
        }
    """
    if model is None:                    # If no model is provided, we cannot infer.
        return None                      # Early exit is clearer than raising here.

    # 1) Pick device by asking any parameter where it lives; default to CPU otherwise.
    device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device("cpu")

    # 2) Prepare input tokens. If caller didn't provide x, we synthesize a random batch of size 1.
    if x is None:
        x = torch.randint(0, model.vocab, (1, model.seq_len), device=device)  # (1,T) random ids
    else:
        if x.dim() == 1:
            x = x.unsqueeze(0)                          # Guarantee a batch dimension → (1,T)
        x = x.to(device)                                 # Move tokens to the same device as the model

    # 3) Evaluation mode for layers like dropout; inference_mode disables grad and is fast/safe.
    was_training = getattr(model, "training", False)     # Remember state to restore after
    model.eval()                                         # Set eval mode
    with torch.inference_mode():
        logits, X = model(x)                             # Forward once: (B,T,V) and (B,T)

        # 4) Decode tokens according to the chosen strategy
        if strategy == "greedy":                         # Deterministic: pick the largest logit at each position
            y = logits.argmax(dim=-1)                   # (B,T)
        elif strategy == "sample":                       # Stochastic: sample with temperature and optional top‑k/top‑p
            y = _sample_from_logits(logits, temperature=temperature, top_k=top_k, top_p=top_p)
        else:
            raise ValueError(f"unknown strategy: {strategy!r}")  # Guard against typos

    # 5) Restore training mode if we changed it (polite behavior in larger programs)
    if was_training:
        model.train()

    # 6) Return both the discrete tokens and the continuous ledger coordinates
    return {"tokens": y, "ledger": X}
