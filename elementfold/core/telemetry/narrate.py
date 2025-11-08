"""
core/telemetry/narrate.py — Human-readable narratives 🗣️
Turns standardized telemetry events into short, readable lines.
"""

from __future__ import annotations
from typing import Dict, Any
from .metrics import validate_event

def narrate_event(evt: Dict[str, Any]) -> str:
    """Return a concise English sentence for a telemetry event."""
    if not validate_event(evt):
        return "⚠️ malformed event"

    e = evt["event"]
    p = evt["payload"]
    # Order roughly mirrors the events we emit most often
    if e == "🏭 factory.start":
        return f"factory started — {p.get('cores', 0)} core(s)"
    if e == "⛔ factory.stop":
        return "factory stopped"
    if e == "🏗️ core.registered":
        return f"core registered — {p.get('core','?')}"
    if e == "🔌 device.attached":
        return f"device attached — {p.get('core','?')}"
    if e == "🧲 device.detached":
        return f"device detached — {p.get('core','?')}"
    if e == "▶️ runtime.start":
        return f"runtime start — {p.get('core','?')} [{p.get('mode','?')}]"
    if e == "⏹ runtime.stop":
        return f"runtime stop — {p.get('core','?')}"
    if e == "🩺 runtime.step":
        return f"{p.get('core','?')} t={p.get('t',0):.3f} ({p.get('mode','?')})"
    if e == "⚙️ runtime.params":
        return f"{p.get('core','?')} params updated {p.get('params',{})}"
    if e == "🎚 mode.change":
        return f"{p.get('core','?')} mode → {p.get('mode','?')}"
    if e == "📸 factory.snapshot":
        return f"snapshot — {p.get('cores',0)} core(s)"

    # default
    return f"{e} {p}"
