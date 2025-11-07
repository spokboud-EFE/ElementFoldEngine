# ElementFold · utils/bootstrap.py
# ============================================================
# Simulation bootstrap utility
# ------------------------------------------------------------
# Purpose:
#   • Create a run directory and default config if missing.
#   • Optionally prompt for physical parameters (λ, D, steps).
#   • Optionally verify remote monitor or dashboard connection.
#   • Never writes outside the run folder; no global side effects.
# ============================================================

from __future__ import annotations
import os, json, sys, time
from typing import Dict, Optional

try:
    from .display import info, warn, success, section
except Exception:  # fallback for early bootstrap
    def info(x: str): print(x)
    def warn(x: str): print("WARN:", x)
    def success(x: str): print("OK:", x)
    def section(x: str): print(f"\n== {x} ==")

from .config import Config
from .io import ensure_dir, write_json


# ============================================================
# 1. Helpers
# ============================================================

def _isatty() -> bool:
    try:
        return sys.stdin.isatty()
    except Exception:
        return False


def _ask(prompt: str, default: Optional[str] = None) -> str:
    if not _isatty():
        return default or ""
    try:
        s = input(prompt).strip()
        return s if s else (default or "")
    except EOFError:
        return default or ""


def _confirm(prompt: str, default: bool = False) -> bool:
    if not _isatty():
        return default
    suffix = " [Y/n]:" if default else " [y/N]:"
    s = _ask(prompt + suffix, None).lower()
    if s == "" and default:
        return True
    return s in {"y", "yes"}


# ============================================================
# 2. Main entry
# ============================================================

def bootstrap_sim_env(path: str = "runs/default", interactive: bool = True) -> Config:
    """
    Initialize a simulation directory and return its Config.

    Behavior:
      • If path/config.json exists → load it.
      • Else create a default config interactively or with defaults.
      • Return Config instance ready for Engine or simulate_once().
    """
    section("Simulation Environment Bootstrap")

    cfg_path = os.path.join(path, "config.json")
    ensure_dir(cfg_path)

    if os.path.exists(cfg_path):
        info(f"Found existing config at {cfg_path}")
        cfg = Config.load(cfg_path)
        return cfg

    if not interactive or not _isatty():
        cfg = Config()
        cfg.save(cfg_path)
        success(f"Created default config at {cfg_path}")
        return cfg

    info("No configuration found; let's create one interactively.")
    lam = float(_ask("λ (letting-go rate) [0.33]: ", "0.33") or 0.33)
    D = float(_ask("D (diffusion strength) [0.15]: ", "0.15") or 0.15)
    steps = int(_ask("Steps per run [100]: ", "100") or 100)
    grid = int(_ask("Grid size N×N [64]: ", "64") or 64)
    phi_inf = float(_ask("Baseline Φ∞ [0.0]: ", "0.0") or 0.0)

    cfg = Config(lambda_=lam, D=D, steps=steps, grid=(grid, grid), phi_inf=phi_inf)
    cfg.save(cfg_path)
    success(f"Saved configuration to {cfg_path}")
    info(json.dumps(cfg.to_dict(), indent=2))

    # Optional: connect to remote dashboard
    if _confirm("Attempt to contact remote monitor/dashboard?", default=False):
        try:
            from urllib.request import urlopen
            url = os.environ.get("ELEMENTFOLD_MONITOR", "http://localhost:8080/health")
            info(f"Pinging {url} ...")
            with urlopen(url, timeout=5) as r:  # nosec B310 (stdlib only)
                if 200 <= r.getcode() < 300:
                    success("Dashboard reachable")
                else:
                    warn(f"HTTP {r.getcode()}")
        except Exception as e:
            warn(f"Monitor unreachable: {e}")

    return cfg


# ============================================================
# 3. CLI
# ============================================================

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "runs/default"
    cfg = bootstrap_sim_env(path)
    print("\n✅ Ready. Configuration summary:")
    print(json.dumps(cfg.to_dict(), indent=2))
