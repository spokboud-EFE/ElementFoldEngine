# ElementFold · cli.py
# ============================================================
# ElementFold Command Line Interface
#
# Friendly entry point for non-experts.
# Commands:
#   train            → run training loop (with progress & saving)
#   infer            → run inference (from checkpoint or quick train)
#   studio           → open interactive steering Studio
#   doctor           → print environment diagnostics
#   steering-train   → train the tiny SteeringController on synthetic data
#
# Example:
#   python -m elementfold train --steps 200 --print-every 50 --out runs/test
# ============================================================

from __future__ import annotations
import argparse, json, os
from pathlib import Path
from typing import Any, Dict, Tuple

from .config import Config
from ..core.runtime import Engine
from ..core.tokenizer import SimpleTokenizer

# pretty tables when available
try:
    from rich import print as rprint
    from rich.table import Table
    _HAS_RICH=True
except Exception:
    _HAS_RICH=False

# ============================================================
# Helpers
# ============================================================

def _parse_caps(s:str|None)->Tuple[int,...]:
    """Parse capacities like '2,6,10,14' → (2,6,10,14)."""
    if not s: return (2,6,10,14)
    try:
        vals=tuple(int(x.strip()) for x in s.split(",") if x.strip())
        if not vals: raise ValueError
        return vals
    except Exception as e:
        raise argparse.ArgumentTypeError("capacities must be comma-separated ints (e.g. '2,6,10,14')") from e

def _coerce_device(choice:str|None)->str|None:
    """Map CLI label to training-loop expectation."""
    if choice in (None,"auto"): return None
    return choice if choice in ("cpu","cuda") else None

def _cfg_with_extras(cfg:Config,**extras:Any)->Config:
    """Patch Config.to_kwargs() so late-added CLI flags reach train_loop."""
    base=cfg.to_kwargs() if hasattr(cfg,"to_kwargs") else dict(vars(cfg))
    extras={k:v for k,v in extras.items() if v is not None}
    def _merged_to_kwargs(_base=base,_extras=extras):
        merged=dict(_base); merged.update(_extras); return merged
    cfg.to_kwargs=_merged_to_kwargs  # type: ignore[attr-defined]
    for k,v in extras.items(): setattr(cfg,k,v)
    return cfg

def _load_config_file(path:str|None)->Dict[str,Any]:
    """Load defaults from TOML or JSON config file."""
    if not path: return {}
    p=Path(path)
    if not p.exists(): raise FileNotFoundError(f"config file not found: {path}")
    if p.suffix.lower()==".json": return json.loads(p.read_text())
    if p.suffix.lower()==".toml":
        try: import tomllib
        except Exception as e: raise RuntimeError("TOML requires Python 3.11+") from e
        return tomllib.loads(p.read_text())
    raise ValueError(f"unsupported config format: {path}")

def _apply_config_defaults(ns:argparse.Namespace,cfg_dict:Dict[str,Any],keys:Tuple[str,...])->None:
    """Apply defaults from config only when CLI flag was absent."""
    for k in keys:
        if getattr(ns,k,None) is None and k in cfg_dict:
            setattr(ns,k,cfg_dict[k])

def _print_kv(title:str, kv:Dict[str,Any])->None:
    """Pretty-print dict as a table when Rich is present."""
    if _HAS_RICH:
        table=Table(title=title)
        table.add_column("Key",style="bold"); table.add_column("Value")
        for k,v in kv.items(): table.add_row(str(k),str(v))
        rprint(table)
    else:
        print(title); [print(f"  - {k}: {v}") for k,v in kv.items()]

def _preflight_summary(kind:str,cfg:Config)->None:
    """Human-readable summary before execution."""
    caps=getattr(cfg,"capacities",(2,6,10,14))
    kv={
        "device": getattr(cfg,"device",None) or "auto (CUDA if available, else CPU)",
        "steps": getattr(cfg,"steps",200),
        "δ★": getattr(cfg,"delta",0.03),
        "shape": f"d={cfg.d} L={cfg.layers} T={cfg.seq_len} b={cfg.batch} V={cfg.vocab}",
        "fold": getattr(cfg,"fold","grid"),
        "capacities": ",".join(map(str,caps)),
        "data": "DataLoader" if getattr(cfg,"use_data",True) else "synthetic tokens",
    }
    for f in ("rung_intent","rung_band","rung_loss_weight"):
        if hasattr(cfg,f): kv[f]=getattr(cfg,f)
    _print_kv(f"⟲ ElementFold • {kind}", kv)

def _env_report()->Dict[str,Any]:
    """Gather basic environment info."""
    try:
        import torch
        cuda=torch.cuda.is_available()
        dev_count=torch.cuda.device_count() if cuda else 0
        name=torch.cuda.get_device_name(0) if cuda and dev_count>0 else None
        mps=getattr(getattr(torch,"backends",None),"mps",None)
        mps_ok=bool(mps.is_available()) if mps else False
        return {
            "python":f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "torch":getattr(torch,"__version__","unknown"),
            "cuda_available":cuda,"cuda_device_count":dev_count,"cuda_device":name,"mps_available":mps_ok}
    except Exception as e: return {"error":f"env probe failed: {e!r}"}

# ============================================================
# Main CLI dispatcher
# ============================================================

def main()->None:
    fmt=argparse.RawTextHelpFormatter
    p=argparse.ArgumentParser(prog="elementfold",formatter_class=fmt,
        description="ElementFold coherence engine CLI")
    sub=p.add_subparsers(dest="cmd",required=True)

    # --- Train subcommand ---
    pt=sub.add_parser("train",help="Train a model (progress + optional checkpoint)",formatter_class=fmt)
    pt.add_argument("--config",type=str,help="Config file (.toml/.json)")
    pt.add_argument("--dump-config",type=str,help="Write resolved training config")
    pt.add_argument("--device",type=str,choices=["auto","cpu","cuda"],default="auto")
    pt.add_argument("--steps",type=int); pt.add_argument("--print-every",type=int)
    pt.add_argument("--out",type=str)
    pt.add_argument("--seq_len",type=int); pt.add_argument("--vocab",type=int)
    pt.add_argument("--d",type=int); pt.add_argument("--layers",type=int)
    pt.add_argument("--heads",type=int); pt.add_argument("--batch",type=int)
    pt.add_argument("--fold",type=str); pt.add_argument("--delta",type=float)
    pt.add_argument("--capacities",type=str)
    pt.add_argument("--no-data",action="store_true")
    pt.add_argument("--rung-intent",type=str,choices=("stabilize","seek","hold"),default="stabilize")
    pt.add_argument("--rung-target-k",type=int); pt.add_argument("--rung-band",type=float)
    pt.add_argument("--rung-loss-weight",type=float,default=0.0)

    # --- Infer subcommand ---
    pi=sub.add_parser("infer",help="Run inference (greedy or sampling)",formatter_class=fmt)
    pi.add_argument("--ckpt",type=str); pi.add_argument("--prompt",type=str)
    pi.add_argument("--strategy",type=str,choices=("greedy","sample"),default="greedy")
    pi.add_argument("--temperature",type=float,default=1.0)
    pi.add_argument("--top-k",type=int,dest="top_k"); pi.add_argument("--top-p",type=float,dest="top_p")
    pi.add_argument("--config",type=str); pi.add_argument("--device",type=str,choices=["auto","cpu","cuda"],default="auto")
    pi.add_argument("--steps",type=int); pi.add_argument("--print-every",type=int)
    pi.add_argument("--seq_len",type=int); pi.add_argument("--vocab",type=int)
    pi.add_argument("--d",type=int); pi.add_argument("--layers",type=int)
    pi.add_argument("--heads",type=int); pi.add_argument("--batch",type=int)
    pi.add_argument("--fold",type=str); pi.add_argument("--delta",type=float)
    pi.add_argument("--capacities",type=str); pi.add_argument("--no-data",action="store_true")
    pi.add_argument("--out",type=str)
    pi.add_argument("--rung-intent",type=str,choices=("stabilize","seek","hold"),default="stabilize")
    pi.add_argument("--rung-target-k",type=int); pi.add_argument("--rung-band",type=float)
    pi.add_argument("--rung-loss-weight",type=float,default=0.0)

    # --- Studio / Doctor / Steering ---
    sub.add_parser("studio",help="Interactive steering Studio")
    sub.add_parser("doctor",help="Environment diagnostics")
    ps=sub.add_parser("steering-train",help="Train the tiny SteeringController",formatter_class=fmt)
    ps.add_argument("--steps",type=int,default=500)
    ps.add_argument("--lr",type=float,default=1e-3)
    ps.add_argument("--delta",type=float,default=0.030908106561043047)
    ps.add_argument("--batch-size",type=int,default=16,dest="batch_size")
    ps.add_argument("--val-frac",type=float,default=0.1)
    ps.add_argument("--device",type=str,choices=["auto","cpu","cuda"],default="auto")
    ps.add_argument("--print-every",type=int,default=100)
    ps.add_argument("--out",type=str)

    args=p.parse_args()

    # --------------------------------------------------------
    # Dispatch
    # --------------------------------------------------------
    if args.cmd=="doctor":
        rep=_env_report(); _print_kv("ElementFold • environment",rep)
        if not rep.get("cuda_available"): print("Tip: install CUDA-enabled PyTorch for GPU.")
        return

    if args.cmd=="train":
        cfg_dict=_load_config_file(args.config)
        cfg_keys=("steps","seq_len","vocab","d","layers","heads","batch",
                  "fold","delta","capacities","use_data",
                  "rung_intent","rung_target_k","rung_band","rung_loss_weight")
        _apply_config_defaults(args,cfg_dict,cfg_keys)
        device=_coerce_device(args.device)
        cfg=Config(device=device,steps=args.steps or 200,
                   vocab=args.vocab or 256,d=args.d or 128,layers=args.layers or 4,
                   heads=args.heads or 4,seq_len=args.seq_len or 128,fold=args.fold or "grid",
                   delta=args.delta or 0.03,capacities=_parse_caps(args.capacities),
                   batch=args.batch or 32,use_data=not args.no_data)
        cfg=_cfg_with_extras(cfg,print_every=args.print_every,out=args.out,
                             rung_intent=args.rung_intent,rung_target_k=args.rung_target_k,
                             rung_band=args.rung_band,rung_loss_weight=args.rung_loss_weight)
        _preflight_summary("train",cfg)
        Engine(cfg).fit(); return

    if args.cmd=="infer":
        if args.ckpt:
            if not Path(args.ckpt).exists(): raise FileNotFoundError(args.ckpt)
            eng=Engine.from_checkpoint(args.ckpt)
            _preflight_summary("infer (from ckpt)",eng.cfg)
        else:
            cfg_dict=_load_config_file(args.config)
            cfg_keys=("steps","seq_len","vocab","d","layers","heads","batch",
                      "fold","delta","capacities","use_data",
                      "rung_intent","rung_target_k","rung_band","rung_loss_weight")
            _apply_config_defaults(args,cfg_dict,cfg_keys)
            device=_coerce_device(args.device)
            cfg=Config(device=device,steps=args.steps or 200,vocab=args.vocab or 256,
                       d=args.d or 128,layers=args.layers or 4,heads=args.heads or 4,
                       seq_len=args.seq_len or 128,fold=args.fold or "grid",
                       delta=args.delta or 0.03,capacities=_parse_caps(args.capacities),
                       batch=args.batch or 32,use_data=not args.no_data)
            cfg=_cfg_with_extras(cfg,print_every=args.print_every,out=args.out,
                                 rung_intent=args.rung_intent,rung_target_k=args.rung_target_k,
                                 rung_band=args.rung_band,rung_loss_weight=args.rung_loss_weight)
            _preflight_summary("infer (quick train)",cfg)
            eng=Engine(cfg); eng.fit()
        if args.prompt:
            print(eng.steer(prompt=args.prompt,modality="language"))
        else:
            res=eng.infer(x=None,strategy=args.strategy,
                          temperature=args.temperature,top_k=args.top_k,top_p=args.top_p)
            text=SimpleTokenizer().decode(res["tokens"].squeeze(0).tolist())
            print(text)
        return

    if args.cmd=="studio":
        from ..experience.studio import studio_main
        _print_kv("⟲ ElementFold • Studio",{"hint":"Use Ctrl+C to exit; steering tips appear in UI."})
        studio_main(); return

    if args.cmd=="steering-train":
        from ..experience.steering_train import fit_steering
        device=_coerce_device(args.device)
        ctrl=fit_steering(steps=args.steps,lr=args.lr,save_path=args.out,
                          delta=args.delta,batch_size=args.batch_size,
                          val_frac=args.val_frac,device=device,
                          print_every=args.print_every)
        if args.out:
            outp=Path(args.out)
            final=outp if outp.suffix else (outp/"checkpoint.pt")
            print(f"✓ SteeringController saved to {final}")
        else:
            print("✓ SteeringController training done (no save path provided)")

if __name__=="__main__": main()
