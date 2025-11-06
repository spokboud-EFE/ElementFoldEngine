# ElementFold · experience/studio.py
# ──────────────────────────────────────────────────────────────────────────────
# A friendly REPL for steering + decoding with readable prompts, soft exits,
# autoloaded adapters, and tiny human counsel lines after outputs.
#
# You can:
#   • pick an adapter (e.g., /mod resonator) and type plain text commands that the
#     adapter understands (like "help", "status", "hold", "step up 2", ...),
#   • adjust decoding (/greedy, /sample t=... k=... p=...),
#   • toggle fallback generation for multimodal/audio (/simulate, /strict),
#   • enable/tune the relaxation clock (/relax ...),
#   • run a quick /infer,
#   • save/load checkpoints,
#   • inspect current settings via /status.
#
# Notes for non‑experts:
#   • β (“beta”)   → exposure (how boldly structure emerges).
#   • γ (“gamma”)  → damping (how much motion is calmed).
#   • ⛔ (“clamp”) → safety cap (how deep negative gate values can go).
#
from __future__ import annotations

import sys
import re
import os
import json
import time
import threading
import itertools
from typing import Tuple, Any, Dict, Optional

import torch

from ..config import Config
from ..runtime import Engine

# Prefer the narrative display helpers; fall back gracefully to old logging.
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

from .steering import SteeringController
from .adapters.base import AdapterRegistry


# ──────────────────────────────────────────────────────────────────────────────
# Lightweight color + spinner (no external deps)
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
            if self._stop.is_set():
                break
            sys.stdout.write("\r" + dim_txt(ch) + " " + self.text)
            sys.stdout.flush()
            time.sleep(0.08)
        # clear line
        sys.stdout.write("\r" + " " * (len(self.text) + 4) + "\r")
        sys.stdout.flush()

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        time.sleep(0.02)


# ──────────────────────────────────────────────────────────────────────────────
# Relaxation clock (diffusion–decay) defaults + helpers
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
    """Coerce user updates into safe ranges; unknown keys ignored."""
    out = dict(cfg)
    for k, v in (updates or {}).items():
        kn = k.strip().lower()
        try:
            if kn in ("eta", "rho", "lambda", "d", "phi_inf", "eta_path_weight", "dt"):
                fv = float(v)
                if kn in ("eta", "lambda", "d"): fv = max(0.0, fv)
                if kn == "rho": fv = min(max(fv, 0.0), 1.0)
                if kn == "eta_path_weight": fv = min(max(fv, 0.0), 1.0)
                if kn == "dt": fv = max(1e-8, fv)
                kmap = {"d": "D"}.get(kn, kn)
                out[kmap] = fv
            elif kn in ("steps",):
                out["steps"] = max(1, int(v))
        except Exception:
            pass
    return out

def _relax_kv_line(cfg: dict) -> str:
    return (f"eta={cfg['eta']:.3f}, w={cfg['eta_path_weight']:.2f}, rho={cfg['rho']:.2f}, "
            f"λ={cfg['lambda']:.3f}, D={cfg['D']:.3f}, Φ∞={cfg['phi_inf']:.2f}, "
            f"steps={int(cfg['steps'])}, dt={cfg['dt']:.2f}")


# ──────────────────────────────────────────────────────────────────────────────
# Autoload adapters so the registry is never empty
# ──────────────────────────────────────────────────────────────────────────────

def _autoload_adapters() -> None:
    """
    Import adapter modules for their registration side‑effects.
    Keeps Studio UX stable across fresh installs.
    """
    try:
        from .adapters import language   # noqa: F401
    except Exception:
        pass
    try:
        from .adapters import audio      # noqa: F401
    except Exception:
        pass
    try:
        from .adapters import multimodal # noqa: F401
    except Exception:
        pass
    try:
        from .adapters import resonator  # optional; registers "resonator"
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Help / parsing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _help_text() -> str:
    names = ", ".join(AdapterRegistry.names()) or "∅"
    return f"""\
Commands:
/mod <name>|list                 select an adapter (available: {names})
/adapters [reload]               list (or reload) registered adapters
/greedy                          set strategy=greedy
/sample t=<T> k=<K> p=<P>        set strategy=sample and knobs (T∈[0,∞), K≥0, P∈(0,1))
/simulate on|off                 allow synthetic fallbacks for multimodal/audio
/strict on|off                   when on (default), *wait* for real data tensors/paths
/relax [show|on|off|reset|set k=v ...]
                                 relaxation clock (diffusion–decay) controls
/infer                           run quick inference (random seed tokens)
/status                          show current settings
/save <path>                     save checkpoint (weights+cfg)
/load <path>                     load checkpoint (lazy; materialized on first use)
/help                            show this help
/quit | /exit                    leave the studio

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
    """
    Parse typical resonator tick/status lines and produce a one‑liner like:
      “locked well; keep γ high” or “on ridge; crossing likely”.
    """
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
    # p½ specific warnings
    if p_half == p_half:  # finite
        if p_half > 0.45:
            notes.append("on the barrier — small nudges decide direction")
        elif p_half < 0.10 and kap == kap and kap > 0.9:
            notes.append("securely away from barrier")
    return " • ".join(notes) if notes else None

def _print_adapter_output(out: Any) -> None:
    """
    Friendly renderer:
      • strings → print as‑is with a counsel line when recognizable,
      • dicts → pretty JSON + extract “waiting” summaries if present.
    """
    if isinstance(out, str):
        print("→", out)
        tip = _counsel_from_text(out)
        if tip:
            print(cyan_txt("💬 " + tip))
        return

    # If the adapter returns a dict, try to surface a short human line first.
    try:
        if isinstance(out, dict):
            cap = out.get("caption")
            if isinstance(cap, str) and cap.strip():
                print(f"→ {cap}")

            # waiting statuses (for strict multimodal/audio, if adapters supply them)
            for key in ("image", "audio"):
                part = out.get(key, {})
                if isinstance(part, dict) and part.get("status") == "waiting":
                    need = part.get("waiting_for", key)
                    hint = part.get("hint", "")
                    print(yellow_txt(f"⚠ waiting for {need} — {hint}"))

            blob = json.dumps(out, indent=2, ensure_ascii=False)
            print(blob[:4000])
            return
    except Exception:
        pass

    print("→", str(out)[:2000])


# ──────────────────────────────────────────────────────────────────────────────
# Main REPL (with prompt and soft exit)
# ──────────────────────────────────────────────────────────────────────────────

def studio_main() -> None:
    # Load adapters up‑front so the registry is populated.
    _autoload_adapters()

    # Config + Engine (lazy model)
    cfg = Config()
    eng = Engine(cfg)                  # lazy; will train or materialize on demand
    ctrl = SteeringController.load_default(cfg.delta)

    # Defaults
    adapter_name = "language"          # may be overridden via /mod <name>
    strategy = "greedy"
    temperature, top_k, top_p = 1.0, 0, 0.0
    simulate, strict = False, True

    # Diffusion–decay is mandatory by default (can be toggled off).
    relax_enabled = True
    relax_cfg: Dict[str, float | int] = dict(RELAX_DEFAULT)

    # Keep a *stateful* runner for the selected adapter (resonator benefits).
    runner = None
    def _ensure_runner() -> None:
        nonlocal runner
        if runner is None:
            factory = AdapterRegistry.get(adapter_name)
            runner = factory()  # stateful callable: runner(model, prompt, style)

    # Opening banner and hints
    print(banner(cfg.delta, 1.0, 0.5))
    avail = ", ".join(AdapterRegistry.names()) or "∅"
    print(f"↳ adapters: {avail}")
    print("↳ type text (adapter run), or commands like '/mod resonator', '/sample t=0.8 k=40 p=0.95', '/infer'.  Ctrl+C to exit.\n")
    info(f"relax: {'on' if relax_enabled else 'off'}  •  {_relax_kv_line(relax_cfg)}")

    try:
        while True:
            try:
                s = input("» ").strip()
            except EOFError:
                print("\n" + dim_txt("⟲ end of input — goodbye."))
                break

            if not s:
                continue

            # Commands
            if s.startswith("/"):
                if s.startswith("/help"):
                    print(_help_text());  continue

                if s.startswith("/quit") or s.startswith("/exit"):
                    print("bye.");  break

                if s.startswith("/mod"):
                    parts = s.split(None, 1)
                    if len(parts) == 1:
                        print("usage: /mod <adapter-name>  or  /mod list")
                        continue
                    arg = parts[1].strip().lower()
                    if arg == "list":
                        print("↳ adapters:", ", ".join(AdapterRegistry.names()) or "∅")
                        continue
                    # verify adapter exists
                    try:
                        AdapterRegistry.get(arg)
                    except KeyError:
                        error(f"unknown adapter: {arg!r}  (available: {', '.join(AdapterRegistry.names()) or '∅'})")
                        continue
                    adapter_name = arg
                    runner = None  # switch ⇒ rebuild runner to reset state for the new adapter
                    success(f"adapter = {adapter_name}")
                    continue

                if s.startswith("/adapters"):
                    if "reload" in s:
                        _autoload_adapters()
                        success("adapters reloaded")
                    print("↳ adapters:", ", ".join(AdapterRegistry.names()) or "∅")
                    continue

                if s.startswith("/greedy"):
                    strategy = "greedy"
                    success("strategy = greedy")
                    continue

                if s.startswith("/sample"):
                    strategy = "sample"
                    temperature, top_k, top_p = _parse_sample(s)
                    success(f"strategy = sample  t={temperature:g}  k={top_k:d}  p={top_p:g}")
                    continue

                if s.startswith("/simulate"):
                    parts = s.split(None, 1)
                    simulate = _parse_on_off(parts[1] if len(parts) > 1 else "", simulate)
                    success(f"simulate = {'on' if simulate else 'off'}")
                    continue

                if s.startswith("/strict"):
                    parts = s.split(None, 1)
                    strict = _parse_on_off(parts[1] if len(parts) > 1 else "", strict)
                    success(f"strict = {'on' if strict else 'off'}")
                    continue

                # ——— relax command family ——————————————————————————————
                if s.startswith("/relax"):
                    parts = s.split(None, 1)
                    arg = (parts[1].strip() if len(parts) > 1 else "").lower()
                    if arg in {"", "show"}:
                        info(f"relax: {'on' if relax_enabled else 'off'}  •  {_relax_kv_line(relax_cfg)}")
                        continue
                    if arg in {"on", "off"}:
                        relax_enabled = (arg == "on")
                        success(f"relax = {'on' if relax_enabled else 'off'}")
                        continue
                    if arg.startswith("reset"):
                        relax_cfg = dict(RELAX_DEFAULT)
                        success("relax knobs reset to defaults")
                        info(_relax_kv_line(relax_cfg));  continue
                    if arg.startswith("set"):
                        # parse k=v pairs
                        updates = {}
                        for tok in re.findall(r"([A-Za-z_]+)\s*=\s*([-+eE0-9.]+)", arg):
                            updates[tok[0]] = tok[1]
                        relax_cfg = _relax_clean_set(relax_cfg, updates)
                        success("relax knobs updated")
                        info(_relax_kv_line(relax_cfg));  continue
                    warn("usage: /relax [show|on|off|reset|set eta=... rho=... D=... lambda=... steps=... dt=...]")
                    continue

                if s.startswith("/infer"):
                    # Ensure a model exists (show a small spinner if we’re training)
                    if eng.model is None:
                        info("reason: no model — training/materializing a fresh one")
                        with _Spinner("training / materializing model …"):
                            _ = eng.infer(x=None, relax=(relax_cfg if relax_enabled else None))
                        success("model ready")
                    out = eng.infer(
                        x=None,
                        strategy=strategy,
                        temperature=temperature,
                        top_k=(top_k if top_k > 0 else None),
                        top_p=(top_p if 0.0 < top_p < 1.0 else None),
                        relax=(relax_cfg if relax_enabled else None),
                    )
                    y = out["tokens"].squeeze(0).tolist()
                    head = y[:64]
                    print("→ /infer tokens:", head, "…" if len(y) > 64 else "")
                    continue

                if s.startswith("/status"):
                    names = ", ".join(AdapterRegistry.names()) or "∅"
                    dev = str(eng.device)
                    line = (
                        f"⟲ status ⟲"
                        f" adapter={adapter_name}"
                        f" strategy={strategy}"
                        f" T={temperature:g}"
                        f" k={top_k}"
                        f" p={top_p:g}"
                        f" simulate={'on' if simulate else 'off'}"
                        f" strict={'on' if strict else 'off'}"
                        f" device={dev}"
                    )
                    print(line)
                    print(f"↳ adapters: {names}")
                    info(f"relax: {'on' if relax_enabled else 'off'}  •  {_relax_kv_line(relax_cfg)}")
                    continue

                if s.startswith("/save"):
                    parts = s.split(None, 1)
                    if len(parts) < 2:
                        print("usage: /save <path>")
                        continue
                    path = parts[1].strip()
                    eng.save(path)
                    success(f"saved checkpoint → {path}")
                    continue

                if s.startswith("/load"):
                    parts = s.split(None, 1)
                    if len(parts) < 2:
                        print("usage: /load <path>")
                        continue
                    path = parts[1].strip()
                    if not os.path.exists(path):
                        error(f"not found: {path}")
                        continue
                    eng = Engine.from_checkpoint(path)
                    success(f"loaded checkpoint (lazy) ← {path}")
                    continue

                warn("unknown command; /help for options")
                continue

            # Normal line: send to the selected adapter
            # Map steering vector → control parameters (β, γ, ⛔) and a 'style' embedding
            v = ctrl(s)                                   # raw ℝ⁸
            params = SteeringController.to_params(v)      # {'beta','gamma','clamp','style'}
            beta, gamma, clamp = params["beta"], params["gamma"], params["clamp"]

            # Gauges (visual)
            print(gauge('β', beta, 2.0), gauge('γ', gamma, 1.0), gauge('⛔', clamp, 10.0))

            # Narrated rows for non‑experts with concrete advice
            # β — novelty gate
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

            # γ — damping
            if gamma < 0.20:
                param("γ", gamma, meaning="may jitter; damping low",
                      quality="warn", maxv=1.0, advice="raise γ a little")
            elif gamma > 0.90:
                param("γ", gamma, meaning="may over‑damp response",
                      quality="warn", maxv=1.0, advice="lower γ a little")
            else:
                param("γ", gamma, meaning="calm, responsive", quality="good", maxv=1.0)

            # ⛔ — clamp
            if clamp < 2.0:
                param("⛔", clamp, meaning="negative gate clipping too soon",
                      quality="warn", maxv=10.0, advice="increase clamp")
            elif clamp > 8.0:
                param("⛔", clamp, meaning="excessive leeway; monitor stability",
                      quality="warn", maxv=10.0, advice="decrease clamp")
            else:
                param("⛔", clamp, meaning="sane safety window", quality="good", maxv=10.0)

            # Ensure a model exists (train/materialize if needed) — with a tiny spinner
            if eng.model is None:
                info("reason: no model — training/materializing a fresh one")
                with _Spinner("training / materializing model …"):
                    _ = eng.infer(x=None, relax=(relax_cfg if relax_enabled else None))  # triggers lazy fit/materialize
                success("model ready")

            # Prepare prompt for adapters that accept dicts so decode knobs (and relax) flow through
            def _wrap_for_adapter(adapter_name: str, line: str) -> Any:
                if adapter_name in {"multimodal", "audio"}:
                    return {
                        "text": line,
                        "decode": {
                            "strategy": strategy,
                            "temperature": float(temperature),
                            "top_k": (int(top_k) if top_k > 0 else None),
                            "top_p": (float(top_p) if 0.0 < top_p < 1.0 else None),
                            "relax": (relax_cfg if relax_enabled else None),
                        },
                        "simulate": bool(simulate),
                        "strict": bool(strict),
                    }
                return line

            prompt_in = _wrap_for_adapter(adapter_name, s)

            # Run the chosen adapter with a *stateful* runner
            try:
                _ensure_runner()
                out = runner(eng.model, prompt_in, v)     # adapter handles the text/dict line
                _print_adapter_output(out)
            except KeyError:
                error(f"unknown adapter: {adapter_name!r}  (available: {', '.join(AdapterRegistry.names()) or '∅'})")
            except Exception as e:
                # Keep REPL alive; surface the error briefly
                error(f"adapter error: {type(e).__name__}: {e}")

    except KeyboardInterrupt:
        print("\n" + dim_txt("⟲ exit requested — goodbye."))

if __name__ == "__main__":
    studio_main()
