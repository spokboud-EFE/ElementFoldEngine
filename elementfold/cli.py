# ElementFold · cli.py
# A narrated, gentle CLI for non‑experts:
# - train: run the main training loop (with progress & saving)
# - infer: quick decode (from ckpt or train‑then‑infer)
# - studio: open the terminal steering studio
# - doctor: print environment sanity checks (CUDA/CPU/etc.)
# - steering-train: train the tiny SteeringController on synthetic pairs
#
# Examples:
#   Train on CPU with progress and save:
#     python -m elementfold train --device cpu --steps 200 --print-every 50 --out runs/test1
#   Infer from a checkpoint via the language adapter:
#     python -m elementfold infer --ckpt runs/test1/checkpoint.pt --prompt "hello"
#   Read defaults from a TOML config (flags still override):
#     python -m elementfold train --config configs/small.toml --steps 500 --out runs/small
#   Print environment diagnostics:
#     python -m elementfold doctor
#   Train steering controller and save weights:
#     python -m elementfold steering-train --steps 800 --out runs/steering/ctrl.pt

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

# Core project surfaces
from .config import Config
from .runtime import Engine
from .tokenizer import SimpleTokenizer

# Optional niceties (pretty tables/progress). Purely optional: we degrade to plain prints when missing.
try:
    from rich import print as rprint  # noqa
    from rich.table import Table  # noqa
    _HAS_RICH = True
except Exception:
    _HAS_RICH = False

# --------------------------- Helpers ---------------------------

def _parse_caps(s: str | None) -> Tuple[int, ...]:
    """Parse capacities like '2,6,10,14' → (2,6,10,14)."""
    if not s:
        return (2, 6, 10, 14)
    try:
        return tuple(int(x.strip()) for x in s.split(",") if x.strip())
    except Exception as e:
        raise argparse.ArgumentTypeError("capacities must be a comma‑separated list of integers") from e


def _coerce_device(choice: str | None) -> str | None:
    """
    Map CLI choice to the training loop expectation:
      'auto' → None   (let the loop pick CUDA if available, else CPU)
      'cpu'  → 'cpu'
      'cuda' → 'cuda'
    """
    if choice is None or choice == "auto":
        return None
    if choice in ("cpu", "cuda"):
        return choice
    return None


def _cfg_with_extras(cfg: Config, **extras: Any) -> Config:
    """
    Ensure new flags (print_every, out, etc.) reach the training loop even if
    Config.to_kwargs() doesn’t yet include them. We wrap to_kwargs() at runtime.
    """
    base = cfg.to_kwargs() if hasattr(cfg, "to_kwargs") else dict(vars(cfg))
    extras = {k: v for k, v in extras.items() if v is not None}

    def _merged_to_kwargs(_base=base, _extras=extras):
        merged = dict(_base)
        merged.update(_extras)
        return merged

    cfg.to_kwargs = _merged_to_kwargs  # type: ignore[attr-defined]
    for k, v in extras.items():
        setattr(cfg, k, v)
    return cfg


def _load_config_file(path: str | None) -> Dict[str, Any]:
    """
    Load training defaults from a config file.
      - TOML (.toml) via stdlib 'tomllib' (Python 3.11+)
      - JSON (.json) via stdlib 'json'
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    if p.suffix.lower() == ".json":
        return json.loads(p.read_text())
    if p.suffix.lower() == ".toml":
        try:
            import tomllib  # Python 3.11+
        except Exception as e:
            raise RuntimeError("TOML config requested but 'tomllib' is unavailable (requires Python 3.11+).") from e
        return tomllib.loads(p.read_text())
    raise ValueError(f"unsupported config format for {path} (use .toml or .json)")


def _apply_config_defaults(ns: argparse.Namespace, cfg_dict: Dict[str, Any], keys: Tuple[str, ...]) -> None:
    """
    Set defaults from a config file only when the CLI flag was not given.
    The explicit CLI always wins.
    """
    for k in keys:
        if getattr(ns, k, None) is None and k in cfg_dict:
            setattr(ns, k, cfg_dict[k])


def _print_kv(title: str, kv: Dict[str, Any]) -> None:
    """Pretty print key→value, using 'rich' when available."""
    if _HAS_RICH:
        table = Table(title=title, show_lines=False)
        table.add_column("Key", style="bold")
        table.add_column("Value")
        for k, v in kv.items():
            table.add_row(str(k), str(v))
        rprint(table)
    else:
        print(title)
        for k, v in kv.items():
            print(f"  - {k}: {v}")


def _preflight_summary(kind: str, cfg: Config) -> None:
    """Give a short narrative summary before running."""
    # Pull the bits users care about most.
    caps = getattr(cfg, "capacities", (2, 6, 10, 14))
    kv = {
        "device": getattr(cfg, "device", None) or "auto (CUDA if available, else CPU)",
        "steps": getattr(cfg, "steps", 200),
        "δ⋆ (delta)": getattr(cfg, "delta", 0.030908106561043047),
        "shape": f"d={getattr(cfg,'d',128)}  L={getattr(cfg,'layers',4)}  T={getattr(cfg,'seq_len',128)}  b={getattr(cfg,'batch',32)}  V={getattr(cfg,'vocab',256)}",
        "fold": getattr(cfg, "fold", "grid"),
        "capacities": ",".join(map(str, caps)) if isinstance(caps, (tuple, list)) else str(caps),
        "data": "DataLoader" if getattr(cfg, "use_data", True) else "synthetic tokens",
    }
    _print_kv(f"⟲ ElementFold • {kind}", kv)


def _env_report() -> Dict[str, Any]:
    """Collect a small environment report (safe on systems without CUDA)."""
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        cuda_dev_count = torch.cuda.device_count() if cuda_ok else 0
        cuda_name = torch.cuda.get_device_name(0) if cuda_ok and cuda_dev_count > 0 else None
        mps_ok = getattr(getattr(torch, "backends", None), "mps", None)
        mps_ok = bool(mps_ok.is_available()) if mps_ok else False
        return {
            "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "torch": getattr(torch, "__version__", "unknown"),
            "cuda_available": cuda_ok,
            "cuda_device_count": cuda_dev_count,
            "cuda_device": cuda_name,
            "mps_available": mps_ok,
        }
    except Exception as e:
        return {"error": f"environment probe failed: {e!r}"}


# --------------------------- CLI ---------------------------

def main() -> None:
    fmt = argparse.RawTextHelpFormatter
    p = argparse.ArgumentParser(
        prog="elementfold",
        description=(
            "ElementFold coherence engine CLI\n"
            "• Train → save a checkpoint, with progress\n"
            "• Infer → decode either via language adapter (--prompt) or raw tokens\n"
            "• Studio → interactive steering loop in the terminal\n"
            "• Doctor → environment sanity check (CUDA/CPU)\n"
            "• Steering‑train → tiny supervised controller on synthetic pairs"
        ),
        formatter_class=fmt,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # --------------------- Train ---------------------
    pt = sub.add_parser(
        "train",
        help="Train a model (prints progress; optionally save a checkpoint)",
        formatter_class=fmt,
    )
    # Config file (defaults) / dump
    pt.add_argument("--config", type=str, default=None,
                    help="Read defaults from a .toml or .json file (flags still override).")
    pt.add_argument("--dump-config", type=str, default=None,
                    help="Write the resolved training config (JSON).")
    # Run controls
    pt.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto",
                    help="Compute device (default: auto = CUDA if available, else CPU).")
    pt.add_argument("--steps", type=int, default=None,
                    help="Optimization steps (default: 200; or from --config).")
    pt.add_argument("--print-every", type=int, default=None,
                    help="Print progress every N steps (omit for silent loop).")
    pt.add_argument("--out", type=str, default=None,
                    help="Checkpoint output path. If a directory or missing extension, saves to <path>/checkpoint.pt.")
    pt.add_argument("--save", type=str, default=None,
                    help="[Deprecated] Same as --out.")
    pt.add_argument("--resume", type=str, default=None,
                    help="Resume training from an Engine checkpoint (continues with the saved config).")
    # Model/data geometry
    pt.add_argument("--seq_len", type=int, default=None, help="Sequence length.")
    pt.add_argument("--vocab", type=int, default=None, help="Vocabulary size.")
    pt.add_argument("--d", type=int, default=None, help="Model width.")
    pt.add_argument("--layers", type=int, default=None, help="Number of FGN blocks.")
    pt.add_argument("--heads", type=int, default=None, help="Attention heads (parity placeholder).")
    pt.add_argument("--batch", type=int, default=None, help="Batch size.")
    pt.add_argument("--fold", type=str, default=None, help="Fold kind (e.g., grid).")
    pt.add_argument("--delta", type=float, default=None, help="Coherence click δ⋆.")
    pt.add_argument("--capacities", type=str, default=None,
                    help="Seat capacities (comma‑separated, e.g. '2,6,10,14').")
    pt.add_argument("--no-data", action="store_true",
                    help="Disable the DataLoader and use synthetic tokens (fast smoke test).")

    # --------------------- Infer ---------------------
    pi = sub.add_parser(
        "infer",
        help="Run inference (greedy or sampling); with --prompt uses the language adapter",
        formatter_class=fmt,
    )
    pi.add_argument("--ckpt", type=str, default=None,
                    help="Checkpoint to load (Engine.save format). If omitted, we train briefly, then infer.")
    pi.add_argument("--prompt", type=str, default=None,
                    help="Text prompt; routes through the language adapter when set.")
    pi.add_argument("--strategy", type=str, default="greedy", choices=("greedy", "sample"),
                    help="Decode strategy.")
    pi.add_argument("--temperature", type=float, default=1.0,
                    help="Sampling temperature when --strategy=sample.")
    pi.add_argument("--top-k", type=int, default=None, dest="top_k",
                    help="Top‑k sampling cutoff (sampling only).")
    pi.add_argument("--top-p", type=float, default=None, dest="top_p",
                    help="Top‑p (nucleus) sampling cutoff (sampling only).")
    # Quick train‑then‑infer controls if no ckpt
    pi.add_argument("--config", type=str, default=None, help="Optional config (.toml/.json) for the temp training.")
    pi.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto",
                    help="Compute device (default: auto).")
    pi.add_argument("--steps", type=int, default=None, help="Temporary training steps (default: 200).")
    pi.add_argument("--print-every", type=int, default=None, help="Progress during temporary training.")
    pi.add_argument("--seq_len", type=int, default=None, help="Sequence length.")
    pi.add_argument("--vocab", type=int, default=None, help="Vocabulary size.")
    pi.add_argument("--d", type=int, default=None, help="Model width.")
    pi.add_argument("--layers", type=int, default=None, help="FGN blocks.")
    pi.add_argument("--heads", type=int, default=None, help="Attention heads (parity).")
    pi.add_argument("--batch", type=int, default=None, help="Batch size.")
    pi.add_argument("--fold", type=str, default=None, help="Fold kind.")
    pi.add_argument("--delta", type=float, default=None, help="Coherence click δ⋆.")
    pi.add_argument("--capacities", type=str, default=None, help="Seat capacities.")
    pi.add_argument("--no-data", action="store_true", help="Use synthetic tokens (fast smoke test).")
    pi.add_argument("--out", type=str, default=None,
                    help="If training happens here, optionally save the resulting checkpoint (same rules as train --out).")

    # --------------------- Studio ---------------------
    ps = sub.add_parser("studio", help="Interactive steering Studio (terminal)")

    # --------------------- Doctor ---------------------
    pd = sub.add_parser("doctor", help="Environment diagnostics (CUDA/CPU/MPS, Torch versions)")

    # ---------------- Steering controller train ----------------
    ps2 = sub.add_parser(
        "steering-train",
        help="Train the tiny SteeringController on synthetic pairs and optionally save weights",
        formatter_class=fmt,
    )
    ps2.add_argument("--steps", type=int, default=500, help="Optimization steps (default: 500).")
    ps2.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate (default: 1e-3).")
    ps2.add_argument("--delta", type=float, default=0.030908106561043047, help="δ⋆ coherence unit (default ≈0.0309).")
    ps2.add_argument("--batch-size", type=int, default=16, dest="batch_size", help="Mini‑batch size (default: 16).")
    ps2.add_argument("--val-frac", type=float, default=0.1, help="Validation split fraction (default: 0.1).")
    ps2.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto",
                     help="Compute device (default: auto).")
    ps2.add_argument("--print-every", type=int, default=100, help="Print progress every N steps.")
    ps2.add_argument("--out", type=str, default=None, help="Where to save weights (file or directory).")

    # --------------------- Parse ---------------------
    args = p.parse_args()

    # --------------------- Dispatch ---------------------

    if args.cmd == "doctor":
        rep = _env_report()
        _print_kv("ElementFold • environment", rep)
        # nudge
        if rep.get("cuda_available") is False:
            print("Tip: For GPUs, install the matching NVIDIA driver and a CUDA‑enabled PyTorch build.")
        return

    if args.cmd == "train":
        # Apply config defaults (if any), respecting explicit flags.
        cfg_dict = _load_config_file(args.config)
        # keys we accept from config files
        cfg_keys = ("steps", "seq_len", "vocab", "d", "layers", "heads", "batch", "fold", "delta", "capacities", "use_data")
        _apply_config_defaults(args, cfg_dict, cfg_keys)

        out_arg = args.out or args.save
        device = _coerce_device(args.device)

        if args.resume:
            # Continue training from a saved Engine checkpoint (uses saved config)
            if not Path(args.resume).exists():
                raise FileNotFoundError(f"resume checkpoint not found: {args.resume}")
            eng = Engine.from_checkpoint(args.resume)
            # Optionally override runtime extras (print_every/out) on resume
            eng.cfg = _cfg_with_extras(eng.cfg, print_every=args.print_every, out=out_arg)
            _preflight_summary("resume", eng.cfg)
            eng.fit()
            return

        # Build fresh config
        cfg = Config(
            device=device,
            steps=args.steps if args.steps is not None else 200,
            vocab=args.vocab if args.vocab is not None else 256,
            d=args.d if args.d is not None else 128,
            layers=args.layers if args.layers is not None else 4,
            heads=args.heads if args.heads is not None else 4,
            seq_len=args.seq_len if args.seq_len is not None else 128,
            fold=args.fold if args.fold is not None else "grid",
            delta=args.delta if args.delta is not None else 0.030908106561043047,
            capacities=_parse_caps(args.capacities) if args.capacities is not None else (2, 6, 10, 14),
            batch=args.batch if args.batch is not None else 32,
            use_data=not args.no_data,
        )
        cfg = _cfg_with_extras(cfg, print_every=args.print_every, out=out_arg)

        _preflight_summary("train", cfg)

        if args.dump_config:
            Path(args.dump_config).write_text(json.dumps(cfg.to_kwargs(), indent=2))
            print(f"✓ wrote resolved config to {args.dump_config}")

        eng = Engine(cfg)
        eng.fit()
        return

    if args.cmd == "infer":
        if args.ckpt:
            if not Path(args.ckpt).exists():
                raise FileNotFoundError(f"checkpoint not found: {args.ckpt}")
            eng = Engine.from_checkpoint(args.ckpt)
            _preflight_summary("infer (from ckpt)", eng.cfg)
        else:
            # Quick train‑then‑infer path
            cfg_dict = _load_config_file(args.config)
            cfg_keys = ("steps", "seq_len", "vocab", "d", "layers", "heads", "batch", "fold", "delta", "capacities", "use_data")
            _apply_config_defaults(args, cfg_dict, cfg_keys)

            device = _coerce_device(args.device)
            cfg = Config(
                device=device,
                steps=args.steps if args.steps is not None else 200,
                vocab=args.vocab if args.vocab is not None else 256,
                d=args.d if args.d is not None else 128,
                layers=args.layers if args.layers is not None else 4,
                heads=args.heads if args.heads is not None else 4,
                seq_len=args.seq_len if args.seq_len is not None else 128,
                fold=args.fold if args.fold is not None else "grid",
                delta=args.delta if args.delta is not None else 0.030908106561043047,
                capacities=_parse_caps(args.capacities) if args.capacities is not None else (2, 6, 10, 14),
                batch=args.batch if args.batch is not None else 32,
                use_data=not args.no_data,
            )
            cfg = _cfg_with_extras(cfg, print_every=args.print_every, out=args.out)
            _preflight_summary("infer (quick train)", cfg)
            eng = Engine(cfg)
            eng.fit()

        # Language adapter path vs. raw token decode
        if args.prompt:
            out_text = eng.steer(prompt=args.prompt, modality="language")
            print(out_text)
        else:
            result = eng.infer(
                x=None,
                strategy=args.strategy,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
            ids = result["tokens"].squeeze(0).tolist()  # (T,)
            text = SimpleTokenizer().decode(ids)
            print(text)
        return

    if args.cmd == "studio":
        from .experience.studio import studio_main
        _print_kv("⟲ ElementFold • Studio", {"hint": "Use Ctrl+C to exit; steering tips are shown in the UI."})
        studio_main()
        return

    if args.cmd == "steering-train":
        from .experience.steering_train import fit_steering
        device = _coerce_device(args.device)
        # The controller trainer accepts these directly
        ctrl = fit_steering(
            steps=args.steps,
            lr=args.lr,
            save_path=args.out,
            delta=args.delta,
            batch_size=args.batch_size,
            val_frac=args.val_frac,
            device=device,
            print_every=args.print_every,
        )
        if args.out:
            print(f"✓ SteeringController saved to {args.out if Path(args.out).suffix else Path(args.out) / 'checkpoint.pt'}")
        else:
            print("✓ SteeringController training done (no save path provided)")
        return


if __name__ == "__main__":
    main()
