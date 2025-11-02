# ElementFold · export.py
# Portable exporters + checkpoints with clear, non‑expert narration.
# Read this like a toolbox:
#   • save_checkpoint / load_checkpoint  — robust weight I/O with config + metadata
#   • export_torchscript                — trace/script a model to a self‑contained .pt
#   • export_onnx                       — ONNX graph export with friendly names/axes

from __future__ import annotations                       # ↻ forward annotations on older Python
import os, json, time                                    # ✴ filesystem • JSON • timestamp
from typing import Any, Dict, Tuple, Optional            # ✴ light typing
import torch                                             # ✴ tensors • JIT • ONNX hooks


# ———————————————————————————————————————————————————————————————
# Tiny FS helper
# ———————————————————————————————————————————————————————————————

def _ensure_dir(path: str) -> None:                      # ✴ make parent folder if needed
    folder = os.path.dirname(path) or "."
    os.makedirs(folder, exist_ok=True)


# ———————————————————————————————————————————————————————————————
# Checkpoint I/O (portable, self‑describing)
# ———————————————————————————————————————————————————————————————

def save_checkpoint(model: torch.nn.Module, path: str, cfg: Any | None = None, extra: Dict[str, Any] | None = None) -> str:
    """
    Save a model state_dict plus optional config/metadata in one file.
    We move tensors to CPU for portability so the file loads anywhere.
    """
    _ensure_dir(path)                                    # 📁 ensure folder exists
    mod = model.module if hasattr(model, "module") else model  # ✴ unwrap DDP if present
    state = {k: v.detach().cpu() for k, v in mod.state_dict().items()}  # 🧱 CPU weights

    # Convert config to a plain dict if it’s a dataclass/object with to_dict()
    if cfg is None:
        cfg_payload = None
    elif hasattr(cfg, "to_dict"):
        cfg_payload = cfg.to_dict()                      # dataclass → dict
    elif isinstance(cfg, dict):
        cfg_payload = cfg                                # already a dict
    else:
        # Best effort: JSON‑serialize arbitrary config objects
        try:
            cfg_payload = json.loads(json.dumps(cfg, default=lambda o: getattr(o, "__dict__", str(o))))
        except Exception:
            cfg_payload = {"repr": repr(cfg)}            # last‑resort repr

    payload = {                                          # 🗂 self‑describing bundle
        "format": "elementfold.ckpt.v1",                 # format tag for forwards‑compat
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),  # ISO‑like timestamp
        "torch": torch.__version__,                      # environment hint
        "state_dict": state,                             # weights (CPU)
        "config": cfg_payload,                           # optional config snapshot
        "extra": dict(extra or {}),                      # freeform metadata
    }
    torch.save(payload, path)                            # 💾 write atomically
    return path                                          # ↤ for convenience in call chains


def load_checkpoint(path: str, model_ctor, map_location: str | torch.device = "cpu", strict: bool = True
                   ) -> Tuple[torch.nn.Module, Any | None, Dict[str, Any]]:
    """
    Load a checkpoint into a fresh model instance.

    Args:
      path:          file produced by save_checkpoint()
      model_ctor:    zero‑arg callable that returns a *fresh* model instance
      map_location:  where tensors land on load (e.g., 'cpu' or 'cuda')
      strict:        whether to enforce an exact key‑by‑key match

    Returns:
      (model, config, info) where info carries missing/unexpected keys.
    """
    chk = torch.load(path, map_location=map_location)    # 📖 read file (portable)
    model = model_ctor()                                 # 🏗️ fresh model
    result = model.load_state_dict(chk["state_dict"], strict=strict)  # ⟲ load weights

    # Normalize “incompatible keys” into plain lists for easy logging
    missing, unexpected = [], []
    try:
        missing = list(getattr(result, "missing_keys", []))
        unexpected = list(getattr(result, "unexpected_keys", []))
    except Exception:
        pass

    info = {                                             # 🧾 small report for the caller
        "format": chk.get("format", "unknown"),
        "torch": chk.get("torch", "unknown"),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "extra": chk.get("extra", {}),
    }
    return model, chk.get("config", None), info          # ↤ model + config snapshot + report


# ———————————————————————————————————————————————————————————————
# TorchScript export (trace/script, optimize if available)
# ———————————————————————————————————————————————————————————————

def export_torchscript(model: torch.nn.Module, example_inputs, path: str,
                       method: str = "trace", optimize: bool = True) -> torch.jit.ScriptModule:
    """
    Turn a PyTorch module into a self‑contained TorchScript file (.pt).

    Args:
      model:          nn.Module in eval mode (we’ll call .eval() for safety)
      example_inputs: tensor or tuple/list of tensors used for tracing
      path:           output file (.pt)
      method:         'trace' (default) or 'script'
      optimize:       try torch.jit.optimize_for_inference to remove dead graph parts

    Returns:
      The in‑memory ScriptModule that was saved.
    """
    _ensure_dir(path)                                    # 📁 ensure folder
    mod = model.module if hasattr(model, "module") else model
    mod.eval()                                           # 🚦 inference graph
    cpu = mod.to("cpu")                                  # 🌍 portable capture
    with torch.no_grad():                                # ≡ no grads in export
        if method == "script":
            ts = torch.jit.script(cpu)                   # ✴ compile by scripting
        else:
            # Pack single tensor into a tuple for torch.jit.trace when needed.
            ex = example_inputs
            if not isinstance(ex, (tuple, list)):
                ex = (ex,)
            ts = torch.jit.trace(cpu, ex, strict=False)  # ✴ compile by tracing
        # Optional graph optimization (available on most builds)
        if optimize:
            try:
                from torch.jit import optimize_for_inference as _opt
                ts = _opt(ts)
            except Exception:
                pass                                     # silently skip if unsupported
        ts.save(path)                                    # 💾 write TorchScript
        return ts                                        # ↤ return compiled module


# ———————————————————————————————————————————————————————————————
# ONNX export (friendly names + dynamic axes)
# ———————————————————————————————————————————————————————————————

def export_onnx(model: torch.nn.Module, example_inputs, path: str,
                opset: int = 17, dynamic: bool = True) -> str:
    """
    Export a model to ONNX with two named outputs:
        'logits' (B,T,V) and 'ledger' (B,T)

    Args:
      model:          nn.Module in eval mode (we’ll call .eval() and move to CPU)
      example_inputs: input tensor(s) — shape should match real inference
      path:           destination .onnx path
      opset:          ONNX opset version (17 is a safe modern default)
      dynamic:        add dynamic axes for batch/time so it runs on variable lengths

    Returns:
      Path to the saved .onnx file.
    """
    _ensure_dir(path)                                    # 📁 ensure folder
    mod = model.module if hasattr(model, "module") else model
    mod.eval().to("cpu")                                 # 🚦 inference on CPU
    # Normalize example inputs to a tuple
    ex = example_inputs if isinstance(example_inputs, (tuple, list)) else (example_inputs,)

    # Names for readability and easier wiring in downstream runtimes
    in_names = [f"input{i}" for i in range(len(ex))]     # e.g., ['input0']
    out_names = ["logits", "ledger"]                     # matches (logits, X) from our model

    # Dynamic axes mapping so ONNX runtimes accept varying batch/time
    dyn = None
    if dynamic:
        dyn = {name: {0: "batch"} for name in in_names}  # batch axis on all inputs
        # Assume input0 is (B,T); propagate dynamic time to outputs
        dyn[in_names[0]][1] = "time"
        dyn["logits"] = {0: "batch", 1: "time"}          # (B,T,V)
        dyn["ledger"] = {0: "batch", 1: "time"}          # (B,T)

    with torch.no_grad():                                # ≡ export is pure forward
        torch.onnx.export(
            mod,                                         # model
            ex,                                          # example inputs
            path,                                        # file path
            input_names=in_names,                        # friendly names
            output_names=out_names,                      # friendly names
            dynamic_axes=dyn,                            # allow variable batch/time
            opset_version=int(opset),                    # ONNX dialect
            do_constant_folding=True,                    # fold statics where safe
        )
    return path                                          # ↤ for call‑chain convenience
