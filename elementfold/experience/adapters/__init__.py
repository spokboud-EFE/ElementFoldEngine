# ElementFold · experience/adapters/__init__.py
# Keep the registry base importable, and import side-effect adapters so Studio sees them.
from .base import AdapterRegistry  # re-export for convenience

# Auto-register the custom adapters on package import:
try:
    from . import resonator  # noqa: F401
except Exception:
    # Adapters are optional; failure here should not crash core functionality.
    pass