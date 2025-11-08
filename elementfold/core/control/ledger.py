"""
core/control/ledger.py — The Chronicle 📖 of ElementFold

Every heartbeat (δ★) leaves a trace here.

──────────────────────────────────────────────────────────────────────
──────────────────────────────────────────────────────────────────────
• Each core keeps a Ledger — its personal memory.
• Every entry is a moment of its relaxation: β exposure, γ damping, κ coherence.
• The Ledger writes in two tongues: data for machines, stories for humans.
• Telemetry listens; the Studio reads aloud.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ====================================================================== #
# 📄 A single line in the chronicle
# ====================================================================== #
@dataclass
class LedgerEntry:
    """Structured record of one Runtime step (one δ★ click)."""

    step_id: int
    t: float
    dt: float
    mode: str
    phase: float
    kappa: float
    params: Dict[str, Any]
    notes: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to a JSON-serializable dictionary."""
        d = asdict(self)
        d["timestamp_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.timestamp))
        return d


# ====================================================================== #
# 🧾 Ledger — persistent record of a core’s evolution
# ====================================================================== #
class Ledger:
    """
    Runtime ledger storing chronological entries.

    - Records each step with contextual narrative.
    - Emits Unicode telemetry for Studio readability.
    - Can autosave to disk in JSONL format for long runs.
    """

    def __init__(
        self,
        name: str = "default",
        *,
        path: Optional[Path] = None,
        autosave: bool = False,
        telemetry: Optional[Any] = None,
        maxlen: Optional[int] = 10_000,
    ) -> None:
        self.name = name
        self.entries: List[LedgerEntry] = []
        self.autosave = autosave
        self.telemetry = telemetry
        self.path = Path(path) if path else None
        self.maxlen = maxlen
        self._counter = 0

        if self.path and autosave:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                f.write("")  # reset file

    # ------------------------------------------------------------------ #
    # ✍️ Record a new step
    # ------------------------------------------------------------------ #
    def record_step(self, runtime, dt: float) -> None:
        """
        Append one step entry from the given runtime.
        Called by Factory after each δ★ tick.
        """
        self._counter += 1
        s = runtime.state
        entry = LedgerEntry(
            step_id=self._counter,
            t=s.t,
            dt=dt,
            mode=s.mode,
            phase=s.phase,
            kappa=s.kappa,
            params=dict(s.params),
            notes=self._compose_note(s),
        )

        self.entries.append(entry)
        if self.maxlen and len(self.entries) > self.maxlen:
            self.entries.pop(0)  # trim oldest entries

        self._emit(entry)
        if self.autosave and self.path:
            self._append_to_file(entry)

    # ------------------------------------------------------------------ #
    # 🪶 Compose plain-language note
    # ------------------------------------------------------------------ #
    def _compose_note(self, state) -> str:
        """
        Generate a short, human-readable sentence summarizing the step.
        Example: "β 1.00 — γ 0.50 — κ 0.97 — harmony stable"
        """
        β = state.params.get("beta")
        γ = state.params.get("gamma")
        κ = state.kappa

        parts: List[str] = []
        if β is not None:
            parts.append(f"β {β:.2f}")
        if γ is not None:
            parts.append(f"γ {γ:.2f}")
        parts.append(f"κ {κ:.2f}")

        sentence = " — ".join(parts)
        # Add mood markers 🌤️ / 🌧️ based on coherence
        if κ > 0.95:
            sentence += " — harmony stable 🌤️"
        elif κ < 0.7:
            sentence += " — turbulence rising 🌧️"
        else:
            sentence += " — breathing normally 🌫️"
        return sentence

    # ------------------------------------------------------------------ #
    # 💾 Persistence
    # ------------------------------------------------------------------ #
    def _append_to_file(self, entry: LedgerEntry) -> None:
        """Append one entry as JSONL to disk."""
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        except (OSError, IOError) as exc:
            print(f"[ledger] write failed: {exc}")

    # ------------------------------------------------------------------ #
    # 📡 Telemetry emission
    # ------------------------------------------------------------------ #
    def _emit(self, entry: LedgerEntry) -> None:
        """Send a concise status message to telemetry."""
        if self.telemetry is None:
            return
        msg = f"#{entry.step_id:04d} ⏱ t={entry.t:.3f} | {entry.notes}"
        try:
            self.telemetry.emit("🪶 ledger.entry", step=entry.step_id, msg=msg)
        except AttributeError:
            print("[ledger] telemetry missing emit()")
        except Exception as exc:
            print(f"[ledger] telemetry error: {exc}")

    # ------------------------------------------------------------------ #
    # 📊 Summaries and utilities
    # ------------------------------------------------------------------ #
    def latest(self) -> Optional[LedgerEntry]:
        """Return the latest entry, or None if empty."""
        return self.entries[-1] if self.entries else None

    def summary(self, n: int = 5) -> str:
        """Return a readable multi-line summary of the last n steps."""
        tail = self.entries[-n:]
        lines = [
            f"#{e.step_id:04d} t={e.t:.3f} dt={e.dt:.3g} {e.notes}"
            for e in tail
        ]
        return "\n".join(lines)

    def export(self) -> List[Dict[str, Any]]:
        """Return all entries as plain dictionaries."""
        return [e.to_dict() for e in self.entries]

    def save(self, path: Optional[Path] = None) -> Path:
        """Write all entries to disk (JSONL)."""
        out = path or self.path or Path(f"{self.name}_ledger.jsonl")
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                for e in self.entries:
                    f.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")
        except (OSError, IOError) as exc:
            print(f"[ledger] save failed: {exc}")
        return out

    # ------------------------------------------------------------------ #
    # 🧹 Maintenance
    # ------------------------------------------------------------------ #
    def clear(self) -> None:
        """Erase all in-memory entries."""
        self.entries.clear()
        self._counter = 0
        self._emit_manual("🧹 ledger.cleared")

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.entries)

    def __iter__(self):
        yield from self.entries

    def __repr__(self) -> str:
        return f"<Ledger name={self.name!r} entries={len(self.entries)}>"

    # ------------------------------------------------------------------ #
    # 🕊️ Internal helper for manual events
    # ------------------------------------------------------------------ #
    def _emit_manual(self, event: str) -> None:
        """Send manual messages to telemetry (for housekeeping)."""
        if self.telemetry is None:
            return
        try:
            self.telemetry.emit(event)
        except Exception as exc:
            print(f"[ledger] telemetry notice failed: {exc}")
