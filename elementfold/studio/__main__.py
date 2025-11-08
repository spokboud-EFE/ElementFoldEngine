"""
elementfold/studio/__main__.py — Studio Entry Point 🎛️
──────────────────────────────────────────────────────────────────────
Starts the persistent hierarchical Studio menu.
No fake runs, no automatic panels — only the control shell.
──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import sys
from elementfold.studio.studio_menu import StudioMenu


def main() -> None:
    """Launch the persistent Studio hierarchical menu."""
    try:
        menu = StudioMenu()
        menu.run()  # the numbered interactive shell
    except KeyboardInterrupt:
        print("\n\033[2m[studio] ✨ interrupted — system cooled and exited\033[0m")
        sys.exit(0)
    except Exception as exc:
        print(f"[main] fatal error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
