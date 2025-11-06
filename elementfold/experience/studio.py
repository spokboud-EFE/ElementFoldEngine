# ElementFold · experience/studio.py
# ──────────────────────────────────────────────────────────────────────────────
# Studio ⇄ UI kernel
# A friendly REPL *and* an event-emitting kernel for adapters, pilot, decoding,
# and a reliability‑gated learning pilot that can take over β/γ/⛔ past a threshold.
#
# Studio goals:
#   • Prefer the local brain; remote brain is a secondary.
#   • Show *all* adapters one per row with numeric selection.
#   • Provide sub‑menus fed by AdapterSpec/Meta where available.
#   • Let a reliability‑gated pilot learn from I/O and—past threshold—take over.
#   • Keep the menu as a fallback; default path remains “type intent → go”.
#   • Emit structured NDJSON events for any UI that wants to mirror Studio.
#
# UI integration (events):
#   • Env:   ELEMENTFOLD_STUDIO_EVENTS=<path or FIFO>
#   • REPL:  /ui [show|on path=...|off|echo on|off]
#
from __future__ import annotations

import sys
import re
import os
import json
import time
import threading
import itertools
import pkgutil
import importlib
import math
from typing import Tuple, Any, Dict, Optional, List, Callable

import torch

from ..utils.bootstrap import bootstrap_brain_env
from ..config import Config
from ..runtime import Engine

# Prefer narrative display helpers; fall back to plain logging.
try:
    from ..utils.display import banner, gauge, info, success, warn, error, param
except Exception:
    from ..utils.logging import banner, gauge  # type: ignore

    def _plain(s: str) -> str: return s
    def info(s: str) -> None:    print(_plain(s))
    def success(s: str) -> None: print(_plain(s))
    def warn(s: str) -> None:    print(_plain(s))
    def error(s: str) -> None:   print(_plain(s))
    def param(name: str, value: float, *, meaning: str, quality: str = "info",
              maxv: float | None = None, advice: str | None = None) -> None:
        q = {"good": "✓", "warn": "!", "bad": "✖"}.get(quality, "•")
        rng = f" [0..{maxv:.2f}]" if maxv is not None else ""
        tip = f"  → {advice}" if advice else ""
        print(f"{q} {name}={value:.2f} — {meaning}{rng}{tip}")

# Steering + optional pilot (graceful shim if pilot absent)
from .steering import SteeringController
try:
    from .steering import SteeringPilot, PilotConfig  # type: ignore
    _HAS_PILOT = True
except Exception:
    _HAS_PILOT = False
    class PilotConfig:  # minimal shim
        def __init__(self, *, threshold: float = 0.80, window: int = 32, min_events: int = 16,
                     rails: Optional[Dict[str, Tuple[float, float]]] = None):
            self.threshold = float(threshold)
            self.window = int(window)
            self.min_events = int(min_events)
            self.rails = rails or {"beta": (0.5, 2.0), "gamma": (0.0, 0.9), "clamp": (1.0, 10.0)}
    class _PilotState:
        def __init__(self): self.reliability, self.events = 0.0, 0
    class SteeringPilot:     # no‑op shim with same surface
        def __init__(self, ctrl: SteeringController, cfg: PilotConfig):
            self.ctrl, self.cfg, self.state = ctrl, cfg, _PilotState()
        def suggest(self, params: Dict[str, float], metrics: Dict[str, float]) -> Dict[str, float]:
            return dict(beta=params["beta"], gamma=params["gamma"], clamp=params["clamp"])
        def observe(self, text: str, params: Dict[str, float], metrics: Dict[str, float], accepted: Optional[bool]):
            self.state.events = min(self.state.events + 1, self.cfg.min_events - 1)
            self.state.reliability = min(self.state.events / max(1, self.cfg.min_events), 0.79)
        def can_take_over(self) -> bool: return False

from .adapters.base import AdapterRegistry, AdapterSpec

# ──────────────────────────────────────────────────────────────────────────────
# Lightweight color + spinner
# ──────────────────────────────────────────────────────────────────────────────

_TTY = sys.stdout.isatty()

def _c(s: str, code: str) -> str:
    return f"\x1b[{code}m{s}\x1b[0m" if _TTY else s

def green_txt(s: str) -> str:  return _c(s, "92")
def yellow_txt(s: str) -> str: return _c(s, "93")
def red_txt(s: str) -> str:    return _c(s, "91")
def cyan_txt(s: str) -> str:   return _c(s, "36")
def dim_txt(s: str) -> str:    return _c(s, "90")

class _Spinner:
    """Minimal console spinner shown while a blocking op runs."""
    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    def __init__(self, text: str = "working…"):
        self.text = text
        self._stop = threading.Event()
        self._thr: Optional[threading.Thread] = None
    def __enter__(self):
        if not _TTY:
            print(self.text + " ...")
            return self
        self._thr = threading.Thread(target=self._run, daemon=True)
        self._thr.start()
        return self
    def _run(self):
        for ch in itertools.cycle(self.FRAMES):
            if self._stop.is_set(): break
            sys.stdout.write("\r" + dim_txt(ch) + " " + self.text)
            sys.stdout.flush()
            time.sleep(0.08)
        sys.stdout.write("\r" + " " * (len(self.text) + 4) + "\r"); sys.stdout.flush()
    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        time.sleep(0.02)

# ──────────────────────────────────────────────────────────────────────────────
# Event emitter (NDJSON sink + callbacks)
# ──────────────────────────────────────────────────────────────────────────────

class StudioEmitter:
    """
    Emit structured events while keeping human prints.
    • Set sink via env ELEMENTFOLD_STUDIO_EVENTS=<path/FIFO> or /ui on path=...
    • Each event is one JSON line: {"t":..., "type":"...", "seq":N, ...}
    """
    def __init__(self, jsonl_path: Optional[str] = None, *, echo: bool = True):
        self._path = jsonl_path
        self._fh = None
        self._echo = bool(echo)
        self._lock = threading.Lock()
        self._seq = 0
        self._cbs: Dict[str, Callable[[Dict[str, Any]], None]] = {}
        if jsonl_path:
            self._open(jsonl_path)

    @classmethod
    def from_env(cls) -> "StudioEmitter":
        path = os.environ.get("ELEMENTFOLD_STUDIO_EVENTS") or os.environ.get("EF_STUDIO_EVENTS")
        echo = os.environ.get("ELEMENTFOLD_STUDIO_ECHO", "1") != "0"
        return cls(path, echo=echo)

    @property
    def echo(self) -> bool: return self._echo

    def set_echo(self, on: bool) -> None:
        self._echo = bool(on)

    def _open(self, path: str) -> None:
        # Open append+line buffered; also works for FIFOs on Linux/Jetson
        try:
            self._fh = open(path, "a", buffering=1, encoding="utf-8")
            self._path = path
        except Exception as e:
            warn(f"events sink open failed: {e}")
            self._fh = None

    def set_sink(self, path: Optional[str]) -> None:
        with self._lock:
            try:
                if self._fh: self._fh.close()
            except Exception:
                pass
            self._fh = None
            self._path = None
            if path:
                self._open(path)

    def attach(self, cb: Callable[[Dict[str, Any]], None]) -> str:
        key = f"cb-{time.time_ns()}"
        self._cbs[key] = cb
        return key

    def detach(self, key: str) -> None:
        self._cbs.pop(key, None)

    def emit(self, typ: str, **payload: Any) -> Dict[str, Any]:
        evt = {"t": time.time(), "type": typ, "seq": self._seq}
        self._seq += 1
        evt.update(payload)
        with self._lock:
            if self._fh:
                try:
                    self._fh.write(json.dumps(evt, ensure_ascii=False) + "\n")
                    self._fh.flush()
                except Exception:
                    pass
        # callbacks best-effort
        for cb in list(self._cbs.values()):
            try: cb(evt)
            except Exception: pass
        return evt

# ──────────────────────────────────────────────────────────────────────────────
# Relaxation defaults + helpers
# ──────────────────────────────────────────────────────────────────────────────

RELAX_DEFAULT: Dict[str, float | int] = {
    "eta": 0.02,             # folds per step (base share rate)
    "eta_path_weight": 0.50, # mix with |ΔX|
    "rho": 0.20,             # lifts sampling temperature with distance
    "lambda": 0.00,          # letting‑go rate
    "D": 0.05,               # diffusion along sequence
    "phi_inf": 0.00,         # calm baseline
    "steps": 1,              # one explicit smoothing tick
    "dt": 1.0,               # step size (small is safe)
}

def _relax_clean_set(cfg: dict, updates: dict) -> dict:
    out = dict(cfg)
    for k, v in (updates or {}).items():
        kn = k.strip().lower()
        try:
            if kn in ("eta", "rho", "lambda", "d", "phi_inf", "eta_path_weight", "dt"):
                fv = float(v)
                if kn in ("eta", "lambda", "d"): fv = max(0.0, fv)
                if kn in ("rho", "eta_path_weight"): fv = min(max(fv, 0.0), 1.0)
                if kn == "dt": fv = max(1e-8, fv)
                kmap = {"d": "D"}.get(kn, kn); out[kmap] = fv
            elif kn in ("steps",): out["steps"] = max(1, int(v))
        except Exception: pass
    return out

def _relax_kv_line(cfg: dict) -> str:
    return (f"eta={cfg['eta']:.3f}, w={cfg['eta_path_weight']:.2f}, rho={cfg['rho']:.2f}, "
            f"λ={cfg['lambda']:.3f}, D={cfg['D']:.3f}, Φ∞={cfg['phi_inf']:.2f}, "
            f"steps={int(cfg['steps'])}, dt={cfg['dt']:.2f}")

# ──────────────────────────────────────────────────────────────────────────────
# Adapter autoload + specs list
# ──────────────────────────────────────────────────────────────────────────────

def _autoload_adapters(verbose: bool = False) -> List[str]:
    imported: List[str] = []
    try:
        from . import adapters as adapters_pkg
    except Exception as e:
        warn(f"failed to import adapters package: {e}")
        return imported
    for _, name, ispkg in pkgutil.iter_modules(adapters_pkg.__path__):
        if name.startswith("_") or name in {"base", "__init__", "loaders"} or ispkg:
            continue
        fq = f"{adapters_pkg.__name__}.{name}"
        try:
            importlib.import_module(fq)
            imported.append(name)
        except Exception as e:
            warn(f"adapter import failed: {name}: {e}")
    if verbose:
        if imported: info("adapters discovered: " + ", ".join(imported))
        else:        warn("no adapters discovered")
    return imported

def _adapter_specs() -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for nm in AdapterRegistry.names():
        try:
            fac = AdapterRegistry.get(nm)
        except KeyError:
            continue
        # Prefer AdapterSpec/Meta; fallback to older attributes/docstrings
        meta = getattr(fac, "__adapter_meta__", None)
        if meta is not None:
            kind = getattr(meta, "kind", "—") or "—"
            desc = getattr(meta, "what", "") or getattr(meta, "why", "") or "No description."
        else:
            kind = getattr(fac, "KIND", None) or getattr(fac, "kind", None) or "—"
            desc = (getattr(fac, "DESCRIPTION", None)
                    or getattr(fac, "description", None)
                    or (fac.__doc__ or "").strip()
                    or "No description.")
        if isinstance(desc, str):
            desc = desc.replace("\n", " ")
            dot = desc.find(".")
            if 20 <= dot <= 140: desc = desc[:dot + 1]
            desc = desc[:160]
        specs.append({"name": nm, "kind": str(kind), "what": str(desc)})
    specs.sort(key=lambda d: d["name"])
    return specs

# ──────────────────────────────────────────────────────────────────────────────
# Help / parsing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _help_text(avail_names: str) -> str:
    return f"""\
Commands:
/menu                           show the numeric top menu (fallback mode)
/mod <name>|list                select an adapter (available: {avail_names})
/adapters [list|reload|json|select <name|#>|info <name|#>]
                                show/pick adapters; `info` shows spec (if provided)
/greedy                         set strategy=greedy
/sample t=<T> k=<K> p=<P>       set strategy=sample and knobs (T∈[0,∞), K≥0, P∈(0,1))
/simulate on|off                allow synthetic fallbacks for multimodal/audio
/strict on|off                  when on (default), wait for real data tensors/paths
/relax [show|on|off|reset|set k=v ...]
                                relaxation clock (diffusion–decay) controls
/params                         list key parameters with human explanations
/pilot [show|on|off|reset|set threshold=... window=... min_events=...
       | set beta_lo=... beta_hi=... gamma_lo=... gamma_hi=... clamp_lo=... clamp_hi=...]
                                reliability‑gated learning pilot
/ui [show|on path=...|off|echo on|off]
                                event stream (NDJSON) for external UIs
/infer                          run quick inference (random seed tokens)
/status                         show current settings
/save <path>                    save checkpoint (weights+cfg)
/load <path>                    load checkpoint (lazy; materialized on first use)
/help                           show this help
/quit | /exit                   leave the studio
"""

def _parse_sample(cmd: str) -> Tuple[float, int, float]:
    m_t = re.search(r"\bt\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))", cmd, re.I)
    m_k = re.search(r"\bk\s*=\s*(\d+)", cmd, re.I)
    m_p = re.search(r"\bp\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))", cmd, re.I)
    T = float(m_t.group(1)) if m_t else 1.0
    K = int(m_k.group(1)) if m_k else 0
    P = float(m_p.group(1)) if m_p else 0.0
    if K < 0: K = 0
    if not (0.0 < P < 1.0): P = 0.0
    if T < 0.0: T = 0.0
    return T, K, P

def _parse_on_off(s: str, default: bool) -> bool:
    s = (s or "").strip().lower()
    if s in {"on", "true", "1", "yes"}:  return True
    if s in {"off", "false", "0", "no"}: return False
    return default

# ──────────────────────────────────────────────────────────────────────────────
# Friendly rendering + small “counsellor” for non‑experts
# ──────────────────────────────────────────────────────────────────────────────

_PHASE_RE = re.compile(r"phase=([A-Z]+)")
_KAP_RE   = re.compile(r"[κk]appa?\s*=\s*([0-9.]+)")
_P12_RE   = re.compile(r"(?:p½|p_half)\s*=\s*([0-9.]+)")

def _counsel_from_text(s: str) -> Optional[str]:
    m_phase = _PHASE_RE.search(s)
    m_kap   = _KAP_RE.search(s)
    m_p12   = _P12_RE.search(s)
    phase = (m_phase.group(1) if m_phase else "?")
    try: kap = float(m_kap.group(1)) if m_kap else float("nan")
    except Exception: kap = float("nan")
    try: p_half  = float(m_p12.group(1)) if m_p12 else float("nan")
    except Exception: p_half = float("nan")
    notes = []
    if phase == "LOCKED":
        if kap >= 0.85 and (p_half == p_half and p_half < 0.15):
            notes.append("locked well; stable and quiet")
        else:
            notes.append("near a rung; damping keeps the rhythm calm")
    elif phase == "MID":
        notes.append("leaning toward the ridge; keep β nimble, γ low")
    elif phase == "CROSSING":
        notes.append("crossing ridge; expect a brief wobble")
    elif phase == "CAPTURE":
        notes.append("re‑locking; raise γ a touch until κ recovers")
    if p_half == p_half:
        if p_half > 0.45: notes.append("on the barrier — small nudges decide direction")
        elif p_half < 0.10 and kap == kap and kap > 0.9: notes.append("securely away from barrier")
    return " • ".join(notes) if notes else None

def _print_adapter_output(out: Any) -> None:
    if isinstance(out, str):
        print("→", out)
        tip = _counsel_from_text(out)
        if tip: print(cyan_txt("💬 " + tip))
        return
    try:
        if isinstance(out, dict):
            cap = out.get("caption")
            if isinstance(cap, str) and cap.strip(): print(f"→ {cap}")
            for key in ("image", "audio"):
                part = out.get(key, {})
                if isinstance(part, dict) and part.get("status") == "waiting":
                    need = part.get("waiting_for", key); hint = part.get("hint", "")
                    print(yellow_txt(f"⚠ waiting for {need} — {hint}"))
            blob = json.dumps(out, indent=2, ensure_ascii=False)
            print(blob[:4000]);  return
    except Exception:
        pass
    print("→", str(out)[:2000])

# ──────────────────────────────────────────────────────────────────────────────
# Params presenter
# ──────────────────────────────────────────────────────────────────────────────

_PARAM_WHAT = {
    "β": ("exposure", "how boldly structure emerges", "raise to surface new patterns"),
    "γ": ("damping", "how much motion is calmed", "raise to steady; lower to respond faster"),
    "⛔": ("safety clamp", "caps negative gate depth", "raise if clipping; lower to be conservative"),
}
_RELAX_WHAT = {
    "eta": ("share rate", "folds per step added along a path", "raises redshift per unit distance"),
    "eta_path_weight": ("path weight", "mix with |ΔX| for adaptive sharing", "more weight → more path‑aware smoothing"),
    "rho": ("temp lift", "adds heat with distance", "avoids collapse in long paths"),
    "lambda": ("let‑go", "local exponential decay toward Φ∞", "higher → faster local calming"),
    "D": ("diffusion", "spreads tension along sequence", "smooths bumps within outputs"),
    "phi_inf": ("calm baseline", "target potential value", "far future background level"),
    "steps": ("ticks", "explicit smoothing ticks per decode", "more steps → smoother, slower"),
    "dt": ("tick size", "size of each step", "smaller is safer"),
}

def _print_params(beta: float, gamma: float, clamp: float, relax_cfg: dict) -> None:
    for name, (tag, does, why) in _PARAM_WHAT.items():
        val = beta if name == "β" else gamma if name == "γ" else clamp
        print(f"{name:>2}  — {tag:14} | now {val:6.3f} | {does} | why: {why}")
    for k, (tag, does, why) in _RELAX_WHAT.items():
        v = relax_cfg[k]
        print(f"{k:>12} — {tag:14} | now {v:8.3f} | {does} | why: {why}")

# ──────────────────────────────────────────────────────────────────────────────
# Small math helpers to reconstruct a raw ℝ⁸ when pilot takes over
# ──────────────────────────────────────────────────────────────────────────────

def _logit(p: float, eps: float = 1e-6) -> float:
    p = max(min(p, 1 - eps), eps)
    return math.log(p / (1 - p))

def _inv_sigmoid_range(y: float, lo: float, hi: float) -> float:
    p = (y - lo) / max(1e-12, (hi - lo))
    return _logit(p)

def _v_from_params(base_v: torch.Tensor, *, beta: float, gamma: float, clamp: float,
                   rails: Dict[str, Tuple[float, float]]) -> torch.Tensor:
    v = base_v.clone().to(dtype=torch.float32)
    b_lo, b_hi = rails["beta"];  g_lo, g_hi = rails["gamma"];  c_lo, c_hi = rails["clamp"]
    v[0] = _inv_sigmoid_range(beta,  b_lo, b_hi)
    v[1] = _inv_sigmoid_range(gamma, g_lo, g_hi)
    v[2] = _inv_sigmoid_range(clamp, c_lo, c_hi)
    return v

# ──────────────────────────────────────────────────────────────────────────────
# Pilot helpers
# ──────────────────────────────────────────────────────────────────────────────

def _pilot_status_line(enabled: bool, pilot: Optional[SteeringPilot]) -> str:
    if not enabled or pilot is None: return "pilot: off"
    st = getattr(pilot, "state", None);  cfg = getattr(pilot, "cfg", None)
    if st is None or cfg is None: return "pilot: on"
    ready = "yes" if hasattr(pilot, "can_take_over") and pilot.can_take_over() else "no"
    reliability = getattr(st, "reliability", 0.0);  events = getattr(st, "events", 0)
    min_events = getattr(cfg, "min_events", 0);     threshold = getattr(cfg, "threshold", 0.0)
    return f"pilot: on  •  reliability={reliability:.2f}  events={events}/{min_events}  ready={ready}  thresh={threshold:.2f}"

def _parse_kv_updates(s: str) -> Dict[str, float]:
    upd = {}
    for k, v in re.findall(r"([A-Za-z_]+)\s*=\s*([-+eE0-9.]+)", s):
        try: upd[k.strip().lower()] = float(v)
        except Exception: pass
    return upd

# ──────────────────────────────────────────────────────────────────────────────
# StudioKernel — shared between REPL and any server host
# ──────────────────────────────────────────────────────────────────────────────

class StudioKernel:
    DEFAULT_RAILS = {"beta": (0.5, 2.0), "gamma": (0.0, 0.9), "clamp": (1.0, 10.0)}

    def __init__(self):
        _autoload_adapters(verbose=False)
        bootstrap_brain_env(interactive=True)

        self.cfg = Config()
        self.eng = Engine(self.cfg)
        self.ctrl = SteeringController.load_default(self.cfg.delta)

        # Rails from controller if available
        self.rails = getattr(self.ctrl, "rails_from", None)
        self.rails = self.rails() if callable(self.rails) else dict(self.DEFAULT_RAILS)

        # Pilot
        self.pilot_enabled = False
        self.pilot = SteeringPilot(self.ctrl, PilotConfig(rails=self.rails)) if SteeringPilot else None

        # Decode defaults
        self.adapter_name = "language"
        self.strategy = "greedy"
        self.temperature, self.top_k, self.top_p = 1.0, 0, 0.0
        self.simulate, self.strict = False, True

        # Relax defaults
        self.relax_enabled = True
        self.relax_cfg: Dict[str, float | int] = dict(RELAX_DEFAULT)

        # Runner & menu context
        self.runner = None
        self.menu_ctx: Optional[Dict[str, Any]] = None

        # Events
        self.em = StudioEmitter.from_env()

    # ——— UX helpers ————————————————————————————————————————————————
    def open_banner(self) -> None:
        avail = ", ".join(AdapterRegistry.names()) or "∅"
        print(banner(self.cfg.delta, 1.0, 0.5))
        print(f"↳ adapters: {avail}")
        print("↳ type text (adapter run), or commands like '/mod resonator', '/sample t=0.8 k=40 p=0.95', '/infer'.  Ctrl+C to exit.\n")
        info(f"relax: {'on' if self.relax_enabled else 'off'}  •  {_relax_kv_line(self.relax_cfg)}")
        info(_pilot_status_line(self.pilot_enabled, self.pilot))
        self.em.emit("studio.ready",
                     delta=self.cfg.delta,
                     adapters=AdapterRegistry.names(),
                     relax=dict(self.relax_cfg),
                     pilot=dict(enabled=self.pilot_enabled))

    def ensure_runner(self) -> None:
        if self.runner is None:
            factory = AdapterRegistry.get(self.adapter_name)
            self.runner = factory()

    def show_adapters_list(self) -> None:
        specs = _adapter_specs()
        if not specs:
            print("↳ adapters: ∅"); self.em.emit("adapters.list", items=[]); return
        print("↳ adapters (pick with '/adapters select <name|#>' or type a number next):")
        rows = []
        for i, sp in enumerate(specs, 1):
            mark = green_txt("●") if sp["name"] == self.adapter_name else "○"
            kind = f"[{sp['kind']}]" if sp["kind"] and sp["kind"] != "—" else ""
            desc = sp["what"]
            print(f"  {i:>2}. {mark} {sp['name']} {kind} — {desc}")
            rows.append({"idx": i, "name": sp["name"], "kind": sp["kind"], "desc": sp["what"]})
        self.em.emit("adapters.list", items=rows, current=self.adapter_name)

    def _spec_to_dict(self, spec: Optional[AdapterSpec]) -> Dict[str, Any]:
        if not spec:
            return {"wait": "allow_sim", "expects": {}, "predicts": {}}
        out = {"wait": spec.wait, "expects": {}, "predicts": dict(spec.predicts or {})}
        for k, ts in (spec.expects or {}).items():
            out["expects"][k] = {
                "shape": getattr(ts, "shape", ()),
                "dtype": getattr(ts, "dtype_str")() if hasattr(ts, "dtype_str") else None,
                "device": getattr(ts, "device_str")() if hasattr(ts, "device_str") else None,
                "doc": getattr(ts, "doc", ""),
            }
        return out

    def show_adapter_info(self, pick: str) -> None:
        if pick.isdigit():
            idx = int(pick) - 1
            choices = _adapter_specs()
            if 0 <= idx < len(choices): pick = choices[idx]["name"]
            else: error("no such adapter number"); return
        try:
            spec = AdapterRegistry.spec(pick)
            fac = AdapterRegistry.get(pick)
        except KeyError:
            error(f"unknown adapter: {pick!r}  (available: {', '.join(AdapterRegistry.names()) or '∅'})")
            return

        print(green_txt(f"adapter: {pick}"))
        kind = getattr(fac, "KIND", None) or getattr(fac, "kind", None) or "—"
        desc = (getattr(fac, "DESCRIPTION", None) or getattr(fac, "description", None) or (fac.__doc__ or "").strip())
        print(f"  kind: {kind}")
        if desc: print(f"  about: {desc.splitlines()[0][:200]}")

        if spec is None or not getattr(spec, "expects", None):
            print("  spec: (none declared)")
            self.em.emit("adapters.info", name=pick, kind=kind, about=desc, spec=None)
            return

        print("  expects:")
        for k, ts in spec.expects.items():
            try:
                shp = getattr(ts, "shape_str")(); dt = getattr(ts, "dtype_str")(); dv = getattr(ts, "device_str")()
                doc = getattr(ts, "doc", "") or ""
                print(f"    • {k:10s} {shp:>12s}  dtype={dt:<8s}  device={dv:<4s}" + (f"  — {doc}" if doc else ""))
            except Exception:
                print(f"    • {k}")
        preds = getattr(spec, "predicts", None) or {}
        if preds:
            print("  predicts:")
            for k, doc in preds.items(): print(f"    • {k}: {doc}")
        wait = getattr(spec, "wait", "allow_sim"); print(f"  wait mode: {wait}")
        self.em.emit("adapters.info", name=pick, kind=kind, about=desc, spec=self._spec_to_dict(spec))

    # ——— menus ————————————————————————————————————————————————————————
    def show_top_menu(self) -> None:
        items = [
            {"idx": 1, "label": "Adapters", "action": "menu:adapters"},
            {"idx": 2, "label": "Decoding", "action": "menu:decoding"},
            {"idx": 3, "label": "Relax", "action": "menu:relax"},
            {"idx": 4, "label": "Pilot", "action": "menu:pilot"},
            {"idx": 5, "label": "Status & Params", "action": "menu:status"},
            {"idx": 6, "label": "Save/Load", "action": "menu:io"},
            {"idx": 7, "label": "Quit", "action": "menu:quit"},
        ]
        print("↳ menu (type a number):")
        for it in items:
            print(f"  {it['idx']:>2}. {it['label']}")
        self.menu_ctx = {"type": "top", "choices": items}
        self.em.emit("menu.show", path="top", items=items)

    def _handle_top_pick(self, n: int) -> None:
        route = {
            1: lambda: (self.show_adapters_list(), self._set_ctx("adapters")),
            2: lambda: (print("  decoding: 1) greedy  2) sample …"), self._set_ctx("decoding")),
            3: lambda: (print("  relax: /relax show | set k=v …"), self._set_ctx("relax")),
            4: lambda: (print("  pilot: /pilot show | on/off | set …"), self._set_ctx("pilot")),
            5: lambda: (self.handle_command("/status"), None),
            6: lambda: (print("  io: /save <path> | /load <path>"), self._set_ctx("io")),
            7: lambda: (_raise_quit(), None),
        }
        fn = route.get(n);  fn and fn()

    def _set_ctx(self, name: str) -> None:
        self.menu_ctx = {"type": name, "choices": []}
        self.em.emit("menu.context", path=name)

    # ——— Command handlers ————————————————————————————————————————————————
    def handle_command(self, s: str) -> bool:
        avail_names = ", ".join(AdapterRegistry.names()) or "∅"

        if s.startswith("/help"):
            print(_help_text(avail_names));  self.em.emit("help.show");  return True

        if s.startswith("/menu"):
            self.show_top_menu();  return True

        if s.startswith("/quit") or s.startswith("/exit"):
            print("bye.");  self.em.emit("studio.exit");  raise KeyboardInterrupt

        # legacy /mod
        if s.startswith("/mod"):
            parts = s.split(None, 1)
            if len(parts) == 1:
                print("usage: /mod <adapter-name>  or  /mod list");  return True
            arg = parts[1].strip().lower()
            if arg == "list":
                self.show_adapters_list()
                self.menu_ctx = {"type": "adapters", "choices": _adapter_specs()}
                return True
            try:
                AdapterRegistry.get(arg)
            except KeyError:
                error(f"unknown adapter: {arg!r}  (available: {avail_names})");  return True
            self.adapter_name, self.runner = arg, None
            success(f"adapter = {self.adapter_name}")
            self.em.emit("adapters.changed", current=self.adapter_name)
            return True

        # /adapters family
        if s.startswith("/adapters"):
            toks = s.split()
            mode = toks[1] if len(toks) > 1 else "list"
            if mode == "reload":
                _autoload_adapters(verbose=True)
                success("adapters reloaded")
                self.show_adapters_list()
                self.menu_ctx = {"type": "adapters", "choices": _adapter_specs()}
                self.em.emit("adapters.reloaded")
                return True
            if mode == "json":
                print(json.dumps(_adapter_specs(), indent=2, ensure_ascii=False));  return True
            if mode == "select":
                if len(toks) < 3:
                    print("usage: /adapters select <name|#>");  return True
                pick = toks[2].strip().lower()
                if pick.isdigit():
                    idx = int(pick) - 1
                    choices = _adapter_specs()
                    if 0 <= idx < len(choices): pick = choices[idx]["name"]
                    else: error("no such adapter number");  return True
                try:
                    AdapterRegistry.get(pick)
                except KeyError:
                    error(f"unknown adapter: {pick!r}  (available: {avail_names})");  return True
                self.adapter_name, self.runner = pick, None
                success(f"adapter = {self.adapter_name}")
                self.em.emit("adapters.changed", current=self.adapter_name)
                return True
            if mode == "info":
                if len(toks) < 3:
                    print("usage: /adapters info <name|#>");  return True
                self.show_adapter_info(toks[2].strip().lower());  return True
            self.show_adapters_list()
            self.menu_ctx = {"type": "adapters", "choices": _adapter_specs()}
            return True

        if s.startswith("/greedy"):
            self.strategy = "greedy";  success("strategy = greedy")
            self.em.emit("decode.strategy", strategy="greedy")
            return True

        if s.startswith("/sample"):
            self.strategy = "sample"
            self.temperature, self.top_k, self.top_p = _parse_sample(s)
            success(f"strategy = sample  t={self.temperature:g}  k={self.top_k:d}  p={self.top_p:g}")
            self.em.emit("decode.strategy", strategy="sample",
                         t=self.temperature, k=self.top_k, p=self.top_p)
            return True

        if s.startswith("/simulate"):
            parts = s.split(None, 1)
            self.simulate = _parse_on_off(parts[1] if len(parts) > 1 else "", self.simulate)
            success(f"simulate = {'on' if self.simulate else 'off'}")
            self.em.emit("decode.simulate", on=self.simulate)
            return True

        if s.startswith("/strict"):
            parts = s.split(None, 1)
            self.strict = _parse_on_off(parts[1] if len(parts) > 1 else "", self.strict)
            success(f"strict = {'on' if self.strict else 'off'}")
            self.em.emit("decode.strict", on=self.strict)
            return True

        # /relax family
        if s.startswith("/relax"):
            parts = s.split(None, 1)
            arg = (parts[1].strip() if len(parts) > 1 else "").lower()
            if arg in {"", "show"}:
                msg = f"relax: {'on' if self.relax_enabled else 'off'}  •  {_relax_kv_line(self.relax_cfg)}"
                info(msg); self.em.emit("relax.status", on=self.relax_enabled, cfg=dict(self.relax_cfg));  return True
            if arg in {"on", "off"}:
                self.relax_enabled = (arg == "on")
                success(f"relax = {'on' if self.relax_enabled else 'off'}")
                self.em.emit("relax.toggle", on=self.relax_enabled);  return True
            if arg.startswith("reset"):
                self.relax_cfg = dict(RELAX_DEFAULT)
                success("relax knobs reset to defaults")
                info(_relax_kv_line(self.relax_cfg))
                self.em.emit("relax.reset", cfg=dict(self.relax_cfg));  return True
            if arg.startswith("set"):
                updates = _parse_kv_updates(arg)
                self.relax_cfg = _relax_clean_set(self.relax_cfg, updates)
                success("relax knobs updated");  info(_relax_kv_line(self.relax_cfg))
                self.em.emit("relax.set", cfg=dict(self.relax_cfg));  return True
            warn("usage: /relax [show|on|off|reset|set eta=... rho=... D=... lambda=... steps=... dt=...]")
            return True

        if s.startswith("/params"):
            v0 = self.ctrl("")
            p0 = SteeringController.to_params(v0)
            _print_params(p0["beta"], p0["gamma"], p0["clamp"], self.relax_cfg)
            self.em.emit("params.report",
                         beta=p0["beta"], gamma=p0["gamma"], clamp=p0["clamp"], relax=dict(self.relax_cfg))
            return True

        # /pilot family
        if s.startswith("/pilot"):
            parts = s.split(None, 1)
            arg = (parts[1].strip() if len(parts) > 1 else "").lower()
            if arg in {"", "show", "status"}:
                line = _pilot_status_line(self.pilot_enabled, self.pilot); info(line)
                self.em.emit("pilot.status", enabled=self.pilot_enabled, line=line);  return True
            if arg in {"on", "off"}:
                self.pilot_enabled = (arg == "on")
                success(f"pilot = {'on' if self.pilot_enabled else 'off'}")
                self.em.emit("pilot.toggle", on=self.pilot_enabled)
                info(_pilot_status_line(self.pilot_enabled, self.pilot));  return True
            if arg.startswith("reset"):
                self.pilot = SteeringPilot(self.ctrl, PilotConfig(rails=self.rails)) if SteeringPilot else None
                success("pilot reset")
                self.em.emit("pilot.reset")
                info(_pilot_status_line(self.pilot_enabled, self.pilot));  return True
            if arg.startswith("set"):
                if not self.pilot:
                    warn("pilot not available");  return True
                updates = _parse_kv_updates(arg)
                if "threshold" in updates: self.pilot.cfg.threshold = float(updates["threshold"])
                if "window" in updates:    self.pilot.cfg.window    = int(updates["window"])
                if "min_events" in updates:self.pilot.cfg.min_events= int(updates["min_events"])
                for k in ("beta","gamma","clamp"):
                    lo_k, hi_k = f"{k}_lo", f"{k}_hi"
                    if lo_k in updates or hi_k in updates:
                        lo = updates.get(lo_k, self.pilot.cfg.rails[k][0])
                        hi = updates.get(hi_k, self.pilot.cfg.rails[k][1])
                        if hi <= lo: warn(f"ignored rails for {k}: hi<=lo")
                        else: self.pilot.cfg.rails[k] = (float(lo), float(hi))
                success("pilot config updated")
                self.em.emit("pilot.set", cfg=dict(threshold=self.pilot.cfg.threshold,
                                                   window=self.pilot.cfg.window,
                                                   min_events=self.pilot.cfg.min_events,
                                                   rails=self.pilot.cfg.rails))
                info(_pilot_status_line(self.pilot_enabled, self.pilot));  return True
            warn("usage: /pilot [show|on|off|reset|set threshold=... window=... min_events=... | set beta_lo=... beta_hi=... gamma_lo=... gamma_hi=... clamp_lo=... clamp_hi=...]")
            return True

        # /ui family (events)
        if s.startswith("/ui"):
            parts = s.split(None, 1)
            arg = (parts[1].strip() if len(parts) > 1 else "").lower()
            if arg in {"", "show", "status"}:
                path = getattr(self.em, "_path", None)
                print(f"ui: sink={'off' if not path else path}  echo={'on' if self.em.echo else 'off'}")
                self.em.emit("ui.status", sink=(path or None), echo=self.em.echo);  return True
            if arg.startswith("on"):
                m = re.search(r"path\s*=\s*([^ \t]+)", arg)
                path = m.group(1) if m else (os.environ.get("ELEMENTFOLD_STUDIO_EVENTS") or "/tmp/elementfold.events.ndjson")
                self.em.set_sink(path);  success(f"ui events → {path}")
                self.em.emit("ui.on", sink=path);  return True
            if arg.startswith("off"):
                self.em.set_sink(None);  success("ui events off")
                self.em.emit("ui.off");  return True
            if arg.startswith("echo"):
                tok = arg.split()
                val = tok[1] if len(tok) > 1 else ""
                on = _parse_on_off(val, self.em.echo)
                self.em.set_echo(on);  success(f"ui echo = {'on' if on else 'off'}")
                self.em.emit("ui.echo", echo=on);  return True
            print("usage: /ui [show|on path=...|off|echo on|off]");  return True

        if s.startswith("/infer"):
            if self.eng.model is None:
                info("reason: no model — training/materializing a fresh one")
                self.em.emit("model.materialize.begin")
                with _Spinner("training / materializing model …"):
                    _ = self.eng.infer(x=None, relax=(self.relax_cfg if self.relax_enabled else None))
                success("model ready");  self.em.emit("model.materialize.done")
            out = self.eng.infer(
                x=None,
                strategy=self.strategy,
                temperature=self.temperature,
                top_k=(self.top_k if self.top_k > 0 else None),
                top_p=(self.top_p if 0.0 < self.top_p < 1.0 else None),
                relax=(self.relax_cfg if self.relax_enabled else None),
            )
            y = out["tokens"].squeeze(0).tolist()
            head = y[:64]
            print("→ /infer tokens:", head, "…" if len(y) > 64 else "")
            self.em.emit("infer.out", head=head, n=len(y))
            return True

        if s.startswith("/status"):
            names = ", ".join(AdapterRegistry.names()) or "∅"
            dev = str(self.eng.device)
            line = (f"⟲ status ⟲ adapter={self.adapter_name} strategy={self.strategy}"
                    f" T={self.temperature:g} k={self.top_k} p={self.top_p:g}"
                    f" simulate={'on' if self.simulate else 'off'} strict={'on' if self.strict else 'off'}"
                    f" device={dev}")
            print(line);  print(f"↳ adapters: {names}")
            info(f"relax: {'on' if self.relax_enabled else 'off'}  •  {_relax_kv_line(self.relax_cfg)}")
            info(_pilot_status_line(self.pilot_enabled, self.pilot))
            self.em.emit("status.report",
                         adapter=self.adapter_name, strategy=self.strategy,
                         T=self.temperature, k=self.top_k, p=self.top_p,
                         simulate=self.simulate, strict=self.strict,
                         device=dev, relax=dict(self.relax_cfg),
                         pilot=dict(enabled=self.pilot_enabled))
            return True

        if s.startswith("/save"):
            parts = s.split(None, 1)
            if len(parts) < 2: print("usage: /save <path>");  return True
            path = parts[1].strip();  self.eng.save(path)
            success(f"saved checkpoint → {path}")
            self.em.emit("ckpt.save", path=path)
            return True

        if s.startswith("/load"):
            parts = s.split(None, 1)
            if len(parts) < 2: print("usage: /load <path>");  return True
            path = parts[1].strip()
            if not os.path.exists(path): error(f"not found: {path}");  return True
            self.eng = Engine.from_checkpoint(path);  self.runner = None
            success(f"loaded checkpoint (lazy) ← {path}")
            self.em.emit("ckpt.load", path=path)
            return True

        return False  # not a command

    # ——— Run a normal adapter turn ————————————————————————————————————
    def run_turn(self, s: str) -> None:
        # Steering vector and params
        v = self.ctrl(s)                                  # ℝ⁸ raw
        params = SteeringController.to_params(v)          # {'beta','gamma','clamp','style'}
        beta, gamma, clamp = params["beta"], params["gamma"], params["clamp"]

        # Gauges + advice (both printed and emitted)
        print(gauge('β', beta, 2.0), gauge('γ', gamma, 1.0), gauge('⛔', clamp, 10.0))
        self.em.emit("steer.gauges", beta=beta, gamma=gamma, clamp=clamp)

        if beta < 0.60:
            param("β", beta, meaning="too cautious; structure under‑exposed",
                  quality="warn", maxv=2.0, advice="step up 1")
        elif beta > 1.85:
            param("β", beta, meaning="very bold; risk instability",
                  quality="bad", maxv=2.0, advice="step down 1")
        elif beta > 1.60:
            param("β", beta, meaning="bold; keep an eye on κ",
                  quality="warn", maxv=2.0, advice="hold or step down 1")
        else:
            param("β", beta, meaning="healthy exposure", quality="good", maxv=2.0)

        if gamma < 0.20:
            param("γ", gamma, meaning="may jitter; damping low",
                  quality="warn", maxv=1.0, advice="raise γ a little")
        elif gamma > 0.90:
            param("γ", gamma, meaning="may over‑damp response",
                  quality="warn", maxv=1.0, advice="lower γ a little")
        else:
            param("γ", gamma, meaning="calm, responsive", quality="good", maxv=1.0)

        if clamp < 2.0:
            param("⛔", clamp, meaning="negative gate clipping too soon",
                  quality="warn", maxv=10.0, advice="increase clamp")
        elif clamp > 8.0:
            param("⛔", clamp, meaning="excessive leeway; monitor stability",
                  quality="warn", maxv=10.0, advice="decrease clamp")
        else:
            param("⛔", clamp, meaning="sane safety window", quality="good", maxv=10.0)

        # Ensure a model exists
        if self.eng.model is None:
            info("reason: no model — training/materializing a fresh one")
            self.em.emit("model.materialize.begin")
            with _Spinner("training / materializing model …"):
                _ = self.eng.infer(x=None, relax=(self.relax_cfg if self.relax_enabled else None))
            success("model ready");  self.em.emit("model.materialize.done")

        # Build adapter prompt
        def _wrap_for_adapter(adapter_name: str, line: str) -> Any:
            if adapter_name in {"multimodal", "audio"}:
                return {
                    "text": line,
                    "decode": {
                        "strategy": self.strategy,
                        "temperature": float(self.temperature),
                        "top_k": (int(self.top_k) if self.top_k > 0 else None),
                        "top_p": (float(self.top_p) if 0.0 < self.top_p < 1.0 else None),
                        "relax": (self.relax_cfg if self.relax_enabled else None),
                    },
                    "simulate": bool(self.simulate),
                    "strict": bool(self.strict),
                }
            return line

        # Pre‑run metrics for the pilot
        pre_metrics = {"pred_beta": beta, "pred_gamma": gamma, "pred_clamp": clamp}
        low, high = self.rails["clamp"]
        pre_metrics["clamp_hit"] = 1.0 if (clamp - low) < 0.5 or (high - clamp) < 0.5 else 0.0

        v_to_use = v
        if self.pilot_enabled and self.pilot is not None:
            suggestion = self.pilot.suggest({"beta": beta, "gamma": gamma, "clamp": clamp}, pre_metrics)
            self.em.emit("pilot.suggest", suggest=dict(suggestion))
            if getattr(self.pilot, "can_take_over", lambda: False)():
                v_to_use = _v_from_params(v, beta=suggestion["beta"], gamma=suggestion["gamma"],
                                           clamp=suggestion["clamp"], rails=self.rails)
                beta, gamma, clamp = suggestion["beta"], suggestion["gamma"], suggestion["clamp"]
                msg = f"pilot takeover → β={beta:.2f}  γ={gamma:.2f}  ⛔={clamp:.2f}"
                info(cyan_txt(msg));  self.em.emit("pilot.takeover", beta=beta, gamma=gamma, clamp=clamp)
            else:
                ghost = f"(pilot would: β→{suggestion['beta']:.2f}  γ→{suggestion['gamma']:.2f}  ⛔→{suggestion['clamp']:.2f})"
                print(dim_txt(ghost));  self.em.emit("pilot.ghost", suggest=dict(suggestion))

        prompt_in = _wrap_for_adapter(self.adapter_name, s)

        # Run adapter
        try:
            self.ensure_runner()
            out = self.runner(self.eng.model, prompt_in, v_to_use)
            # Compact representation for UI
            short = None
            if isinstance(out, str):
                tip = _counsel_from_text(out)
                short = {"text": out[:800], "tip": tip}
            elif isinstance(out, dict):
                short = {
                    "caption": (out.get("caption") if isinstance(out.get("caption"), str) else None),
                    "keys": sorted(list(out.keys()))[:20],
                }
            _print_adapter_output(out)
            self.em.emit("adapter.out", adapter=self.adapter_name, summary=short)
            if self.pilot_enabled and self.pilot is not None:
                self.pilot.observe(s, {"beta": beta, "gamma": gamma, "clamp": clamp}, pre_metrics, accepted=None)
        except KeyError:
            msg = f"unknown adapter: {self.adapter_name!r}  (available: {', '.join(AdapterRegistry.names()) or '∅'})"
            error(msg);  self.em.emit("error", kind="adapter.missing", msg=msg)
        except Exception as e:
            msg = f"adapter error: {type(e).__name__}: {e}"
            error(msg);  self.em.emit("error", kind="adapter.exception", msg=str(e))
            if self.pilot_enabled and self.pilot is not None:
                bad = dict(pre_metrics); bad["clamp_hit"] = 1.0
                self.pilot.observe(s, {"beta": beta, "gamma": gamma, "clamp": clamp}, bad, accepted=False)

# ──────────────────────────────────────────────────────────────────────────────
# REPL
# ──────────────────────────────────────────────────────────────────────────────

def _raise_quit():  # tiny helper for menu route
    raise KeyboardInterrupt

def studio_main() -> None:
    kernel = StudioKernel()
    try:
        kernel.open_banner()
        while True:
            try:
                s = input("» ").strip()
            except EOFError:
                print("\n" + dim_txt("⟲ end of input — goodbye."));  kernel.em.emit("studio.eof");  break
            if not s:
                continue

            # numeric quick‑pick after menu/adapters list
            if kernel.menu_ctx and s.isdigit():
                idx = int(s)
                if kernel.menu_ctx.get("type") == "top":
                    kernel._handle_top_pick(idx);  continue
                if kernel.menu_ctx.get("type") == "adapters":
                    choices = kernel.menu_ctx.get("choices", [])
                    if 1 <= idx <= len(choices):
                        kernel.adapter_name = choices[idx - 1]["name"];  kernel.runner = None
                        success(f"adapter = {kernel.adapter_name}")
                        kernel.show_adapters_list();  kernel.menu_ctx = None
                        kernel.em.emit("adapters.changed", current=kernel.adapter_name)
                        continue
                    else:
                        warn("no such adapter number");  continue

            if s.startswith("/"):
                handled = kernel.handle_command(s)
                if handled: continue
                warn("unknown command; /help for options");  continue

            # Normal line → adapter
            kernel.run_turn(s)

    except KeyboardInterrupt:
        print("\n" + dim_txt("⟲ exit requested — goodbye."));  kernel.em.emit("studio.exit")

# Legacy helper kept for compatibility (invoked by older tooling)
def _show_adapters_list(current: str) -> None:
    specs = _adapter_specs()
    if not specs:
        print("↳ adapters: ∅");  return
    print("↳ adapters (pick with '/adapters select <name|#>' or type a number next):")
    for i, sp in enumerate(specs, 1):
        mark = green_txt("●") if sp["name"] == current else "○"
        kind = f"[{sp['kind']}]" if sp["kind"] and sp["kind"] != "—" else ""
        desc = sp["what"]
        print(f"  {i:>2}. {mark} {sp['name']} {kind} — {desc}")

if __name__ == "__main__":
    studio_main()
