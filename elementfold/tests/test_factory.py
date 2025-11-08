"""
Unit tests for Factory orchestration: registration, synchronization, and entanglement.
"""

from elementfold.core.control.factory import Factory
from elementfold.core.control.runtime import Runtime
from elementfold.core.control.ledger import Ledger


def test_factory_register_and_snapshot():
    factory = Factory()
    rt = Runtime()
    lg = Ledger()
    factory.register_core("alpha", rt, lg)
    snap = factory.snapshot()
    assert "alpha" in snap
    assert isinstance(snap["alpha"], dict)

def test_synchronization_and_entanglement():
    f = Factory()
    rt1, rt2 = Runtime(), Runtime()
    lg1, lg2 = Ledger(), Ledger()
    f.register_core("c1", rt1, lg1)
    f.register_core("c2", rt2, lg2)
    f.synchronize()
    f.entangle()
    assert abs(rt1.phase - rt2.phase) < 1e-9
