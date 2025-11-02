# ElementFold · experience/studio.py
# A minimal, friendly REPL for steering + decoding, now with:
#   • modality switching (language | audio | multimodal),
#   • live steering gauges (β, γ, ⛔) via SteeringController.to_params(),
#   • sampling knobs (strategy, temperature, top-k, top-p),
#   • adapter run or direct /infer,
#   • /status (show current settings), /adapters (list), /save, /load.
#
# Commands:
#   /mod <name>|list       — set or list modalities
#   /adapters              — list registered adapters
#   /greedy                — use greedy decoding for /infer
#   /sample t=<T> k=<K> p=<P>  — sampling knobs
#   /infer                 — run infer (random seed tokens)
#   /status                — print current settings
#   /save <path>           — save checkpoint (weights+cfg)
#   /load <path>           — load checkpoint (lazy materialize)
#   /help                  — show help
#
# Otherwise, any line is treated as a prompt sent to the current adapter via Engine.steer().

from __future__ import annotations
import sys, re, os, json, torch
from ..config import Config
from ..runtime import Engine
from ..utils.logging import banner, gauge
from .steering import SteeringController
from .adapters.base import AdapterRegistry

HELP = """\
Commands:
  /mod <language|audio|multimodal>   set modality
  /mod list                          list available modalities
  /adapters                          list available adapters
  /greedy                            set strategy=greedy
  /sample t=<temperature> k=<top_k> p=<top_p>   set strategy=sample + knobs
  /infer                              run infer (random tokens)
  /status                             print current settings
  /save <path>                        save checkpoint (weights+cfg)
  /load <path>                        load checkpoint (lazy; materialized on first use)
  /help                               show this help
"""

def _parse_sample(cmd: str):
    """Parse '/sample t=... k=... p=...' into floats/ints with defaults."""
    m_t = re.search(r"t\s*=\s*([0-9.]+)", cmd)
    m_k = re.search(r"k\s*=\s*([0-9]+)", cmd)
    m_p = re.search(r"p\s*=\s*([0-9.]+)", cmd)
    t = float(m_t.group(1)) if m_t else 1.0
    k = int(m_k.group(1)) if m_k else 0
    p = float(m_p.group(1)) if m_p else 0.0
    return t, k, p

def _print_status(modality, strategy, temperature, top_k, top_p, eng: Engine):
    adapters = ", ".join(AdapterRegistry.names())
    dev = str(eng.device)
    print(f"⟲ status ⟲  modality={modality}  strategy={strategy}  T={temperature:g}  k={top_k}  p={top_p:g}  device={dev}")
    print(f"↳ adapters: {adapters}")

def studio_main():
    # Config + Engine (lazy model)
    cfg = Config()
    eng = Engine(cfg)           # lazy; will train or materialize on demand
    ctrl = SteeringController.load_default(cfg.delta)

    # Defaults
    modality = "language"
    strategy = "greedy"
    temperature, top_k, top_p = 1.0, 0, 0.0

    print(banner(cfg.delta, 1.0, 0.5))
    print(f"↳ adapters: {', '.join(AdapterRegistry.names())}")
    print("↳ type text (adapter run), or commands like '/mod audio', '/sample t=0.8 k=40 p=0.95', '/infer'.  Ctrl+C to exit.\n")

    for line in sys.stdin:
        s = line.strip()
        if not s:
            continue

        # Commands
        if s.startswith("/"):
            if s.startswith("/help"):
                print(HELP)
                continue

            if s.startswith("/mod"):
                parts = s.split(None, 1)
                if len(parts) == 1:
                    print("usage: /mod language|audio|multimodal  or  /mod list")
                    continue
                arg = parts[1].strip().lower()
                if arg == "list":
                    print("↳ adapters:", ", ".join(AdapterRegistry.names()))
                    continue
                modality = arg
                print(f"✓ modality = {modality}")
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
                _print_status(modality, strategy, temperature, top_k, top_p, eng)
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

        # Adapter run on the typed prompt
        v = ctrl(s)                                   # raw ℝ⁸
        params = SteeringController.to_params(v)      # map → {'beta','gamma','clamp','style'}
        beta, gamma, clamp = params["beta"], params["gamma"], params["clamp"]
        # Gauges
        print(gauge('β', beta, 2.0), gauge('γ', gamma, 1.0), gauge('⛔', clamp, 10.0))

        # Ensure a model is available (train or materialize pending checkpoint)
        if eng.model is None:
            _ = eng.infer(x=torch.randint(0, cfg.vocab, (1, cfg.seq_len)))  # triggers lazy fit/materialize

        # Run adapter
        try:
            adapter = AdapterRegistry.get(modality)
            out = adapter()(eng.model, s, v)
            if isinstance(out, str):
                print("→", out)
            else:
                print("→", json.dumps(out, indent=2)[:2000])
        except KeyError:
            print(f"✖ unknown modality: {modality!r}  (available: {', '.join(AdapterRegistry.names())})")

if __name__ == "__main__":
    studio_main()
