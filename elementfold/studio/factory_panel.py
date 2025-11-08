"""
Factory Panel — Studio-side interface exposing live Factory controls.
Allows users to spawn, synchronize, and entangle cores interactively.
"""

from __future__ import annotations
from elementfold.core.control.factory import Factory


class FactoryPanel:
    """Human-facing control surface for the Factory."""

    def __init__(self, factory: Factory) -> None:
        self.factory = factory

    def show_menu(self) -> None:
        """Display available Factory commands."""
        print("\n🏭  Factory Control Menu")
        print(" 1. Start factory")
        print(" 2. Stop factory")
        print(" 3. Synchronize cores (δ★)")
        print(" 4. Entangle cores (ρ, κ)")
        print(" 5. Snapshot state")
        print(" 0. Exit\n")

    def run(self) -> None:
        """Simple CLI loop."""
        while True:
            self.show_menu()
            choice = input("Select: ").strip()
            if choice == "1":
                self.factory.start()
            elif choice == "2":
                self.factory.stop()
            elif choice == "3":
                self.factory.synchronize()
            elif choice == "4":
                self.factory.entangle()
            elif choice == "5":
                snap = self.factory.snapshot()
                print(f"Snapshot: {list(snap.keys())}")
            elif choice == "0":
                break
            else:
                print("Invalid option.")
