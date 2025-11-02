# ElementFold · cli.py
# A tiny command‑line interface so you can train, infer, or open the Studio
# without writing Python.
#
# Usage examples:
#   • Train and save a checkpoint:
#       python -m elementfold.cli train --steps 500 --save run.ckpt
#   • Quick greedy inference from a checkpoint (language path via adapter):
#       python -m elementfold.cli infer --ckpt run.ckpt --prompt "hello"
#   • Open the interactive Studio (steering UI in the terminal):
#       python -m elementfold.cli studio
#
# Subcommands:
#   train  — run the project’s default training loop; optional --save ckpt
#   infer  — load (or train) and decode; optional --prompt routes through language adapter
#   studio — interactive “intent → steering → adapter” loop

from __future__ import annotations                     # Forward annotations (older Python)
import argparse                                        # CLI argument parsing
import sys                                             # Exit codes
from typing import Tuple                               # Small type hint
from .config import Config                             # Typed configuration carrier
from .runtime import Engine                            # Orchestration spine (fit / infer / steer / save)
from .tokenizer import SimpleTokenizer                 # For decoding raw token outputs when not using the adapter
from .utils.logging import banner                      # Pretty banner for quick sanity checks


def _parse_caps(s: str | None) -> Tuple[int, ...]:
    """Parse capacities like '2,6,10,14' → (2,6,10,14)."""
    if not s:
        return (2, 6, 10, 14)
    try:
        return tuple(int(x.strip()) for x in s.split(",") if x.strip())
    except Exception:
        raise argparse.ArgumentTypeError("capacities must be a comma‑separated list of integers")


def _build_config_from_args(args: argparse.Namespace) -> Config:
    """Turn common CLI flags into a Config instance."""
    cfg = Config(
        device=args.device if hasattr(args, "device") else "cuda",
        steps=args.steps if hasattr(args, "steps") else 200,
        vocab=args.vocab if hasattr(args, "vocab") else 256,
        d=args.d if hasattr(args, "d") else 128,
        layers=args.layers if hasattr(args, "layers") else 4,
        heads=args.heads if hasattr(args, "heads") else 4,
        seq_len=args.seq_len if hasattr(args, "seq_len") else 128,
        fold=args.fold if hasattr(args, "fold") else "grid",
        delta=args.delta if hasattr(args, "delta") else 0.030908106561043047,
        capacities=_parse_caps(getattr(args, "capacities", None)),
        batch=args.batch if hasattr(args, "batch") else 32,
        use_data=not getattr(args, "no_data", False),
    )
    return cfg


def main() -> None:
    # — top‑level parser (we use subparsers for clean UX) —
    p = argparse.ArgumentParser(prog="elementfold", description="ElementFold coherence engine CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # — train subcommand —
    pt = sub.add_parser("train", help="train a model (optionally save a checkpoint)")
    pt.add_argument("--steps", type=int, default=200, help="optimization steps (default: 200)")
    pt.add_argument("--seq_len", type=int, default=128, help="sequence length (default: 128)")
    pt.add_argument("--vocab", type=int, default=256, help="vocab size (default: 256)")
    pt.add_argument("--d", type=int, default=128, help="model width (default: 128)")
    pt.add_argument("--layers", type=int, default=4, help="number of FGN blocks (default: 4)")
    pt.add_argument("--heads", type=int, default=4, help="attention heads (kept for parity; default: 4)")
    pt.add_argument("--batch", type=int, default=32, help="batch size (default: 32)")
    pt.add_argument("--fold", type=str, default="grid", help="fold kind (default: grid)")
    pt.add_argument("--delta", type=float, default=0.030908106561043047, help="coherence click δ⋆")
    pt.add_argument("--capacities", type=str, default="2,6,10,14", help="seat capacities (e.g., '2,6,10,14')")
    pt.add_argument("--device", type=str, default="cuda", help="device: 'cuda' | 'cpu' | 'auto'")
    pt.add_argument("--no-data", action="store_true", help="disable DataLoader, use synthetic tokens")
    pt.add_argument("--save", type=str, default=None, help="optional checkpoint path to save")
    # — infer subcommand —
    pi = sub.add_parser("infer", help="run inference (greedy or sampling); with --prompt uses language adapter")
    pi.add_argument("--ckpt", type=str, default=None, help="optional checkpoint path to load (Engine.save format)")
    pi.add_argument("--prompt", type=str, default=None, help="input text; routes through language adapter when set")
    pi.add_argument("--strategy", type=str, default="greedy", choices=("greedy", "sample"), help="decode strategy")
    pi.add_argument("--temperature", type=float, default=1.0, help="sampling temperature when --strategy=sample")
    pi.add_argument("--top-k", type=int, default=None, dest="top_k", help="top‑k sampling cutoff")
    pi.add_argument("--top-p", type=float, default=None, dest="top_p", help="top‑p (nucleus) sampling cutoff")
    pi.add_argument("--steps", type=int, default=200, help="if no ckpt, steps to train before inferring")
    pi.add_argument("--seq_len", type=int, default=128, help="sequence length (for training if needed)")
    pi.add_argument("--vocab", type=int, default=256, help="vocab size (for training if needed)")
    pi.add_argument("--d", type=int, default=128, help="model width (for training if needed)")
    pi.add_argument("--layers", type=int, default=4, help="FGN blocks (for training if needed)")
    pi.add_argument("--heads", type=int, default=4, help="attention heads (for parity)")
    pi.add_argument("--batch", type=int, default=32, help="batch size (for training if needed)")
    pi.add_argument("--fold", type=str, default="grid", help="fold kind (default: grid)")
    pi.add_argument("--delta", type=float, default=0.030908106561043047, help="coherence click δ⋆")
    pi.add_argument("--capacities", type=str, default="2,6,10,14", help="seat capacities")
    pi.add_argument("--device", type=str, default="auto", help="device: 'cuda' | 'cpu' | 'auto'")
    pi.add_argument("--no-data", action="store_true", help="disable DataLoader, use synthetic tokens")
    # — studio subcommand —
    ps = sub.add_parser("studio", help="interactive steering Studio (terminal)")

    # — parse args and dispatch —
    args = p.parse_args()

    if args.cmd == "train":
        cfg = _build_config_from_args(args)               # Build a Config from flags
        eng = Engine(cfg)                                 # Create an Engine with that config
        model = eng.fit()                                 # Train to completion
        if args.save:                                     # Optionally save a checkpoint
            eng.save(args.save)
            print(f"✓ saved checkpoint → {args.save}")
        else:
            print("✓ training done (no checkpoint path provided)")
        return

    if args.cmd == "infer":
        # If a checkpoint is provided, restore it; else we’ll train with cfg below.
        eng = Engine.from_checkpoint(args.ckpt) if args.ckpt else Engine(_build_config_from_args(args))
        if args.prompt:
            # Route through the language adapter for text‑in → text‑out UX.
            out = eng.steer(prompt=args.prompt, modality="language")
            print(out)
        else:
            # Raw inference path: return tokens+ledger; we decode tokens for display.
            result = eng.infer(x=None, strategy=args.strategy, temperature=args.temperature,
                               top_k=args.top_k, top_p=args.top_p)
            ids = result["tokens"].squeeze(0).tolist()    # (T,) token ids
            text = SimpleTokenizer().decode(ids)          # best‑effort UTF‑8 text
            print(text)
        return

    if args.cmd == "studio":
        # Import on demand so CLI remains light if Studio isn’t used.
        from .experience.studio import studio_main
        studio_main()
        return


if __name__ == "__main__":                                # Allow `python -m elementfold.cli`
    main()                                                 # Dispatch based on CLI args
