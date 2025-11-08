"""
elementfold/studio/__main__.py — Entry point for the live Studio 🎛️

Run the Studio from a terminal:

    python -m elementfold.studio

It launches the StudioMenu with telemetry and brain panels.
"""

from __future__ import annotations
import sys
from elementfold.studio.studio_menu import StudioMenu


def main() -> None:
    """Start the persistent Studio shell."""
    try:
        menu = StudioMenu()
        menu.start()
    except KeyboardInterrupt:
        print("\n[main] Interrupted by user — shutting down...")
        try:
            menu.stop()
        except Exception:
            sys.exit(0)
    except Exception as exc:
        print(f"[main] fatal error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
