# ElementFold · experience/studio.py
# ──────────────────────────────────────────────────────────────────────────────
# A minimal, friendly REPL for steering + decoding.
#
# You can:
#   • pick an adapter (e.g., /mod resonator) and type plain text commands that the
#     adapter understands (like "help", "status", "hold", "step up 2", ...),
#   • adjust decoding (/greedy, /sample t=... k=... p=...),
#   • run a quick /infer,
#   • save/load checkpoints,
#   • inspect current settings via /status.
#
# Notes for non‑experts:
#   • β (“beta”) controls how boldly we let structure emerge.
#   • γ (“gamma”) is damping; larger values calm motion.
#   • ⛔ (“clamp”) is a safety cap for very negative gate values.
#
from __future__ import annotations

import sys
import re
import os
import json
import torch
from typing import Tuple

from ..config import Config
from ..runtime import Engine
from ..utils.logging import banner, gauge
from .steering import SteeringController
from .adapters.base import AdapterRegistry


def _help_text() -> str:
    names = ", ".join(AdapterRegistry.names())
    return f"""\
Commands:
/mod <name>|list                 select an adapter (available: {names})
/adapters                        list registered adapters
/greedy                          set strategy=greedy
/sample t=<T> k=<K> p=<P>        set strategy=sample and knobs (T∈[0,∞), K≥0, P∈(0,1))
/infer                           run quick inference (random seed tokens)
/status                          show current settings
/save <path>                     save checkpoint (weights+cfg)
/load <path>                     load checkpoint (lazy; materialized on first use)
/help                            show this help

Usage:
  1) Choose an adapter:   /mod resonator
  2) Type adapter text:   help   |   init δ=0.5   |   hold   |   step up 2   |   tick 5
  3) Watch gauges:        β (exposure), γ (damping), ⛔ (safety clamp)
"""


def _parse_sample(cmd: str) -> Tuple[float, int, float]:
    """Parse '/sample t=... k=... p=...' into (T, K, P) with sensible defaults."""
    m_t = re.search(r"\bt\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))", cmd, re.I)
    m_k = re.search(r"\bk\s*=\s*(\d+)", cmd, re.I)
    m_p = re.search(r"\bp\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))", cmd, re.I)
    T = float(m_t.group(1)) if m_t else 1.0
    K = int(m_k.group(1)) if m_k else 0
    P = float(m_p.group(1)) if m_p else 0.0
    # Gentle validation / clamping for user comfort
    if K < 0:
        K = 0
    if not (0.0 < P < 1.0):
        P = 0.0  # interpret “no top‑p” outside (0,1)
    if T < 0.0:
        T = 0.0
    return T, K, P


def _print_status(adapter_name: str, strategy: str, temperature: float, top_k: int, top_p: float, eng: Engine) -> None:
    names = ", ".join(AdapterRegistry.names())
    dev = str(eng.device)
    print(f"⟲ status ⟲  adapter={adapter_name}  strategy={strategy}  T={temperature:g}  k={top_k}  p={top_p:g}  device={dev}")
    print(f"↳ adapters: {names}")


def studio_main() -> None:
    # Config + Engine (lazy model)
    cfg = Config()
    eng = Engine(cfg)                  # lazy; will train or materialize on demand
    ctrl = SteeringController.load_default(cfg.delta)

    # Defaults
    adapter_name = "language"          # may be overridden via /mod <name> (e.g., 'resonator')
    strategy = "greedy"
    temperature, top_k, top_p = 1.0, 0, 0.0

    # Opening banner and hints
    print(banner(cfg.delta, 1.0, 0.5))
    print(f"↳ adapters: {', '.join(AdapterRegistry.names())}")
    print("↳ type text (adapter run), or commands like '/mod resonator', '/sample t=0.8 k=40 p=0.95', '/infer'.  Ctrl+C to exit.\n")

    for line in sys.stdin:
        s = line.strip()
        if not s:
            continue

        # Commands
        if s.startswith("/"):
            if s.startswith("/help"):
                print(_help_text())
                continue

            if s.startswith("/mod"):
                parts = s.split(None, 1)
                if len(parts) == 1:
                    print("usage: /mod <adapter-name>  or  /mod list")
                    continue
                arg = parts[1].strip().lower()
                if arg == "list":
                    print("↳ adapters:", ", ".join(AdapterRegistry.names()))
                    continue
                # verify adapter exists
                try:
                    AdapterRegistry.get(arg)
                except KeyError:
                    print(f"✖ unknown adapter: {arg!r}  (available: {', '.join(AdapterRegistry.names())})")
                    continue
                adapter_name = arg
                print(f"✓ adapter = {adapter_name}")
                continue

            if s.startswith("/adapters"):
                print("↳ adapters:", ", ".join(AdapterRegistry.names()))
                continue

            if s.startswith("/greedy"):
                strategy = "greedy"
                print("✓ strategy = greedy")
                continue

            if s.startswith("/sample"):
                strategy = "sample"
                temperature, top_k, top_p = _parse_sample(s)
                print(f"✓ strategy = sample  t={temperature:g}  k={top_k:d}  p={top_p:g}")
                continue

            if s.startswith("/infer"):
                out = eng.infer(
                    x=None,
                    strategy=strategy,
                    temperature=temperature,
                    top_k=(top_k if top_k > 0 else None),
                    top_p=(top_p if 0.0 < top_p < 1.0 else None),
                )
                y = out["tokens"].squeeze(0).tolist()
                head = y[:64]
                print("→ /infer tokens:", head, "…" if len(y) > 64 else "")
                continue

            if s.startswith("/status"):
                _print_status(adapter_name, strategy, temperature, top_k, top_p, eng)
                continue

            if s.startswith("/save"):
                parts = s.split(None, 1)
                if len(parts) < 2:
                    print("usage: /save <path>")
                    continue
                path = parts[1].strip()
                eng.save(path)
                print(f"✓ saved checkpoint → {path}")
                continue

            if s.startswith("/load"):
                parts = s.split(None, 1)
                if len(parts) < 2:
                    print("usage: /load <path>")
                    continue
                path = parts[1].strip()
                if not os.path.exists(path):
                    print(f"✖ not found: {path}")
                    continue
                # Build a new Engine from the checkpoint (lazy materialization).
                eng = Engine.from_checkpoint(path)
                print(f"✓ loaded checkpoint (lazy) ← {path}")
                continue

            print("unknown command; /help for options")
            continue

        # Normal line: send to the selected adapter
        # Map steering vector → control parameters (β, γ, ⛔) and a 'style' embedding
        v = ctrl(s)                                   # raw ℝ⁸
        params = SteeringController.to_params(v)      # {'beta','gamma','clamp','style'}
        beta, gamma, clamp = params["beta"], params["gamma"], params["clamp"]

        # Gauges (just a friendly visual; not authoritative)
        print(gauge('β', beta, 2.0), gauge('γ', gamma, 1.0), gauge('⛔', clamp, 10.0))

        # Ensure a model exists (train/materialize if needed)
        if eng.model is None:
            _ = eng.infer(x=torch.randint(0, cfg.vocab, (1, cfg.seq_len)))  # triggers lazy fit/materialize

        # Run the chosen adapter
        try:
            adapter_factory = AdapterRegistry.get(adapter_name)
            runner = adapter_factory()                 # stateful runner
            out = runner(eng.model, s, v)             # adapter handles the text line
            if isinstance(out, str):
                print("→", out)
            else:
                print("→", json.dumps(out, indent=2)[:2000])
        except KeyError:
            print(f"✖ unknown adapter: {adapter_name!r}  (available: {', '.join(AdapterRegistry.names())})")
        except Exception as e:
            # Keep REPL alive; surface the error briefly
            print(f"✖ adapter error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    studio_main()
