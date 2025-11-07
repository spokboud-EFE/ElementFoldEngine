# ElementFold · experience/studio_brain_panel.py
# ============================================================
# Studio UI Panel — Local Brain Status
# ------------------------------------------------------------
# Periodically queries /brain/loop/status and prints a structured
# dashboard panel with telemetry bars and reasoning narrative.
# ============================================================

from __future__ import annotations
import time, json, urllib.request, urllib.error
from typing import Dict, Any

from ..utils.display import section, param, info, warn, success, GLYPHS

URL = "http://127.0.0.1:8080/brain/loop/status"

def _fetch_status() -> Dict[str, Any]:
    try:
        with urllib.request.urlopen(URL, timeout=3) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.URLError as e:
        warn(f"Cannot reach server: {e}")
    except Exception as e:
        warn(f"Error: {e}")
    return {}

def render_status(d: Dict[str, Any]) -> None:
    section("Local Brain · Runtime Loop")
    if not d:
        warn("No status data available.")
        return

    status = d.get("status", "unknown")
    dt = d.get("seconds_since_last_step")
    decision = d.get("last_decision") or {}
    tele = d.get("telemetry") or {}

    # Loop status line
    loop_line = f"Loop: {status}"
    if dt is not None:
        loop_line += f"  ({dt:.1f}s since last)"
    info(loop_line)

    # Brain reasoning summary
    action = decision.get("action")
    comment = decision.get("comment")
    if action:
        info(f"Action: {action}")
    if comment:
        info(f"Comment: {comment}")

    # Parameters (if available)
    params = decision.get("params") or {}
    if params:
        for k, v in params.items():
            param(k, float(v), meaning="adjusted by brain", quality="info", maxv=1.0)

    # Telemetry display
    if tele:
        param("κ", tele.get("kappa", 0.0), meaning="coherence", quality="good", maxv=1.0)
        param("p½", tele.get("p_half", 0.0), meaning="barrier proximity", quality="neutral", maxv=1.0)
        param("resid_std", tele.get("resid_std", 0.0), meaning="variance", quality="info", maxv=0.1)
        param("margin_mean", tele.get("margin_mean", 0.0), meaning="safety mean", quality="good", maxv=1.0)
        param("margin_min", tele.get("margin_min", 0.0), meaning="worst margin", quality="warn", maxv=1.0)
    success("Panel refreshed.")


def live(refresh: float = 3.0) -> None:
    """
    Continuous live panel loop (press Ctrl+C to stop).
    """
    section("ElementFold Studio — Live Brain Panel")
    try:
        while True:
            d = _fetch_status()
            render_status(d)
            time.sleep(refresh)
            print("\n")
    except KeyboardInterrupt:
        success("Live panel stopped by user.")
