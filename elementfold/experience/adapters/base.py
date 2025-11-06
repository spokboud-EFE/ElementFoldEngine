# ElementFold · experience/adapters/base.py
# ──────────────────────────────────────────────────────────────────────────────
# Adapters are small “bridges” from a model to a user modality (language, audio,
# photonics, …). The registry below maps a *name* → *factory*, where a factory
# is a zero‑arg callable returning a *runner*. A runner has the signature:
#
#     runner(model, prompt, style) -> Any
#
# Public API (tiny, stdlib‑only):
#   • AdapterRegistry.register(name, factory, *, overwrite=True)
#   • AdapterRegistry.get(name)          → factory (KeyError on miss, with hints)
#   • AdapterRegistry.ensure(name, default) → factory (install default if absent)
#   • AdapterRegistry.names()            → sorted tuple of names
#   • AdapterRegistry.has(name) / "name" in AdapterRegistry
#   • AdapterRegistry.register_fn("name")  decorator form
#   • AdapterRegistry.unregister(name)   → remove a factory (testing/dev only)
#   • AdapterRegistry.clear()            → drop all factories (testing/dev only)
#
# Additions (compatible; optional to use):
#   • AdapterSpec / TensorSpec — declare what tensors an adapter *expects*,
#     and what it *predicts*. Plain words: tell Studio what needs to arrive.
#   • with_spec(spec) decorator — attach a spec to a factory (no extra imports).
#   • AdapterRegistry.spec(name) → AdapterSpec | None
#   • AdapterRegistry.instantiate(name) → runner, with spec attached to runner
#   • AdapterRegistry.validate(name, feed) → readiness report for real data
#   • AdapterRegistry.simulate(name, **kw) → optional simulate() if provided
#
# Design notes:
#   • Names are normalized to lowercase (ASCII) for stable CLI UX.
#   • Errors include close‑match suggestions (handy at the REPL).
#   • Thread‑safe writes via a simple lock; reads are uncontended.
#   • Backward‑compatible: existing calls keep working as‑is.
#   • “Wait for real data” flow: adapters can publish *TensorSpec*s that Studio
#     can check before calling the runner; if not ready and the adapter supports
#     simulation, Studio can call simulate() instead.
#
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple
import threading
import warnings
import difflib

try:
    import torch  # optional in this file; only used for dtype/device names
except Exception:  # pragma: no cover
    torch = None  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# Friendly, minimal specs (plain words, light structure)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TensorSpec:
    """
    What a single tensor should look like at the adapter boundary.

    Plain words:
      • shape: tuple of ints where -1 means “any length” (unknown/variable).
               You may also pass symbolic labels like 'B','T','C' for docs.
      • dtype: torch dtype or a string like 'float32' (we compare by name).
      • device: 'cpu' | 'cuda' | 'any'
    """
    shape: Tuple[Any, ...] = field(default_factory=tuple)  # e.g. ('B', 'T') or (None, 16000) or (-1, 16000)
    dtype: Any = None                                      # torch.dtype | 'float32' | None
    device: str = "any"                                    # 'any' | 'cpu' | 'cuda'
    doc: str = ""                                          # one‑line narrative for help panels

    def shape_str(self) -> str:
        def _piece(p: Any) -> str:
            if p is None or p == -1:
                return "?"
            return str(p)
        return "(" + ", ".join(_piece(p) for p in self.shape) + ")"

    def dtype_str(self) -> str:
        if self.dtype is None:
            return "∗"
        try:
            if torch is not None and isinstance(self.dtype, torch.dtype):  # type: ignore[attr-defined]
                return str(self.dtype).replace("torch.", "")
        except Exception:
            pass
        return str(self.dtype)

    def device_str(self) -> str:
        return self.device or "any"


@dataclass(frozen=True)
class AdapterSpec:
    """
    Adapter contract for Studio and tooling.

    Plain words:
      • name: user‑facing key ("resonator", "language", …).
      • expects: which input tensors must be present (by field name).
      • predicts: human‑readable map of outputs the adapter will produce.
      • wait: 'require' means *must* have real data; 'allow_sim' means fall back
              to simulate() if data is missing; 'simulate_only' never waits.
    """
    name: str
    description: str = ""
    expects: Dict[str, TensorSpec] = field(default_factory=dict)
    predicts: Dict[str, str] = field(default_factory=dict)
    wait: str = "allow_sim"  # 'require' | 'allow_sim' | 'simulate_only'

    def pretty(self) -> str:
        if not self.expects:
            return f"{self.name}: expects ∅ (none); wait={self.wait}"
        parts = [f"{self.name}: wait={self.wait}"]
        for k, spec in self.expects.items():
            parts.append(f"  • {k}: {spec.shape_str()}  dtype={spec.dtype_str()}  device={spec.device_str()}"
                         + (f"  — {spec.doc}" if spec.doc else ""))
        if self.predicts:
            parts.append("  ↳ predicts:")
            for k, doc in self.predicts.items():
                parts.append(f"     • {k}: {doc}")
        return "\n".join(parts)


# Small decorator to pin a spec on a factory (no extra machinery needed).
def with_spec(spec: AdapterSpec) -> Callable[[Callable[[], Callable]], Callable[[], Callable]]:
    def _decor(factory: Callable[[], Callable]) -> Callable[[], Callable]:
        setattr(factory, "__adapter_spec__", spec)
        return factory
    return _decor


# ──────────────────────────────────────────────────────────────────────────────
# Readiness checking (duck‑typed; torch optional)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ReadyReport:
    ok: bool
    problems: Tuple[str, ...] = ()
    missing: Tuple[str, ...] = ()
    extras: Tuple[str, ...] = ()
    summary: str = ""


def _dtype_name(dt: Any) -> str:
    try:
        if torch is not None and isinstance(dt, torch.dtype):  # type: ignore[attr-defined]
            return str(dt).replace("torch.", "")
    except Exception:
        pass
    s = str(dt)
    return s.replace("torch.", "")


def _device_type(dev: Any) -> str:
    # Accept torch.device, strings, or objects with '.type'
    if hasattr(dev, "type"):
        return str(getattr(dev, "type"))
    if isinstance(dev, str):
        return dev
    return str(dev)


def _check_one_tensor(name: str, value: Any, spec: TensorSpec) -> Optional[str]:
    # Presence and tensor‑ish properties
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    device = getattr(value, "device", None)

    # shape checks (only when spec carries concrete expectations)
    if spec.shape:
        if shape is None:
            return f"{name}: missing .shape (got {type(value).__name__})"
        if len(shape) != len(spec.shape):
            return f"{name}: rank {len(shape)} ≠ expected {len(spec.shape)} ({spec.shape_str()})"
        for i, (got, want) in enumerate(zip(shape, spec.shape)):
            if want in (None, -1):
                continue  # wildcard
            # Allow symbolic labels like 'B','T' → treated as wildcard for matching
            if isinstance(want, str):
                continue
            if int(got) != int(want):
                return f"{name}: dim[{i}]={int(got)} ≠ {int(want)} (spec {spec.shape_str()})"

    # dtype
    if spec.dtype is not None:
        if dtype is None:
            return f"{name}: missing .dtype"
        if _dtype_name(dtype) != _dtype_name(spec.dtype):
            return f"{name}: dtype={_dtype_name(dtype)} ≠ { _dtype_name(spec.dtype) }"

    # device
    want_dev = (spec.device or "any").lower()
    if want_dev != "any":
        got_dev = _device_type(device).lower()
        if "cuda" in want_dev:
            if "cuda" not in got_dev:
                return f"{name}: device={got_dev} but CUDA requested"
        elif "cpu" in want_dev:
            if "cpu" not in got_dev:
                return f"{name}: device={got_dev} but CPU requested"

    return None  # all good


def validate_feed(feed: Mapping[str, Any], spec: Optional[AdapterSpec]) -> ReadyReport:
    """
    Compare an incoming data feed (a dict‑like) to the adapter's expectations.

    Returns a ReadyReport with ok=True if everything matches or if no spec is
    declared (we don't block legacy adapters).
    """
    if spec is None or not spec.expects:
        return ReadyReport(ok=True, summary="no spec declared (legacy adapter)")

    # Required keys
    missing = tuple(k for k in spec.expects.keys() if k not in feed)
    problems: list[str] = []
    for k, tspec in spec.expects.items():
        if k not in feed:
            continue
        err = _check_one_tensor(k, feed[k], tspec)
        if err:
            problems.append(err)

    # Extraneous keys (not an error; helpful signal)
    extras = tuple(sorted(set(feed.keys()) - set(spec.expects.keys())))

    ok = (len(missing) == 0) and (len(problems) == 0)
    if ok:
        summary = "feed matches expectations"
    else:
        lines = []
        if missing:
            lines.append("missing: " + ", ".join(missing))
        if problems:
            lines.extend(problems)
        if extras:
            lines.append("extras: " + ", ".join(extras))
        summary = "; ".join(lines)

    return ReadyReport(ok=ok,
                       problems=tuple(problems),
                       missing=missing,
                       extras=extras,
                       summary=summary)


# ──────────────────────────────────────────────────────────────────────────────
# Registry (thread‑safe writes; friendly reads)
# ──────────────────────────────────────────────────────────────────────────────

def _norm(name: str) -> str:
    """Normalize external names to a stable, case‑insensitive key."""
    return str(name).strip().lower()


class AdapterRegistry:
    """
    A tiny, thread‑safe mapping from adapter name → adapter factory.

    Definitions:
      • name:    string key like "language", "resonator", "interferometer".
      • factory: zero‑arg callable returning a *runner* callable.
                 Runner signature: runner(model, prompt, style) -> Any
    """

    _reg: Dict[str, Callable[[], Callable]] = {}
    _lock = threading.Lock()

    # ────────────────────────────────────────────────────────────────────────
    # Core operations
    # ────────────────────────────────────────────────────────────────────────

    @classmethod
    def register(
        cls,
        name: str,
        factory: Callable[[], Callable],
        *,
        overwrite: bool = True,
    ) -> None:
        """
        Register an adapter factory under `name`.

        Args:
            name:     registry key (case‑insensitive; stored lowercase).
            factory:  zero‑arg callable producing a runner.
            overwrite: if False and name exists, raise KeyError; otherwise replace
                       and emit a gentle warning so accidental shadowing is visible.

        Raises:
            TypeError: if `factory` is not callable.
            KeyError:  if `overwrite=False` and `name` already exists.
        """
        if not callable(factory):
            raise TypeError(f"adapter factory for {name!r} must be callable, got {type(factory).__name__}")
        key = _norm(name)
        with cls._lock:
            if key in cls._reg and not overwrite:
                raise KeyError(f"adapter:{key} already registered; set overwrite=True to replace")
            if key in cls._reg and overwrite:
                prev = cls._reg[key]
                warnings.warn(
                    f"Adapter '{key}' overwritten (was {getattr(prev, '__name__', type(prev).__name__)}).",
                    RuntimeWarning,
                    stacklevel=2,
                )
            cls._reg[key] = factory

    @classmethod
    def get(cls, name: str) -> Callable[[], Callable]:
        """
        Fetch the adapter factory for `name`.

        Returns:
            The registered zero‑arg factory.

        Raises:
            KeyError with a friendly message and close‑match suggestions.
        """
        key = _norm(name)
        try:
            return cls._reg[key]
        except KeyError:
            avail = sorted(cls._reg.keys())
            hint = ""
            if avail:
                # Suggest up to 3 close matches (threshold tuned for short names)
                matches = difflib.get_close_matches(key, avail, n=3, cutoff=0.6)
                if matches:
                    hint = f"  did you mean: {, ".join(matches)}?"
            available = ", ".join(avail) if avail else "∅ (none registered)"
            raise KeyError(f"adapter:{key} not found; available = [{available}].{hint}") from None

    @classmethod
    def ensure(cls, name: str, default: Callable[[], Callable]) -> Callable[[], Callable]:
        """
        Get existing factory for `name`, or register `default` if absent.

        Returns:
            The existing or newly registered factory.

        Raises:
            TypeError if `default` is not callable.
        """
        if not callable(default):
            raise TypeError(f"default factory for {name!r} must be callable, got {type(default).__name__}")
        key = _norm(name)
        with cls._lock:
            if key not in cls._reg:
                cls._reg[key] = default
            return cls._reg[key]

    # ────────────────────────────────────────────────────────────────────────
    # Convenience and UX
    # ────────────────────────────────────────────────────────────────────────

    @classmethod
    def names(cls) -> Tuple[str, ...]:
        """Return a sorted, immutable tuple of registered adapter names."""
        return tuple(sorted(cls._reg.keys()))

    @classmethod
    def has(cls, name: str) -> bool:
        """Quick membership test (same as `name in AdapterRegistry`)."""
        return _norm(name) in cls._reg

    def __contains__(self, name: str) -> bool:  # Allows: "language" in AdapterRegistry
        return self.has(name)

    def __len__(self) -> bool:  # How many adapters are registered?
        return len(self._reg)

    # ────────────────────────────────────────────────────────────────────────
    # Decorator form for concise registration
    # ────────────────────────────────────────────────────────────────────────

    @classmethod
    def register_fn(cls, name: str) -> Callable[[Callable[[], Callable]], Callable[[], Callable]]:
        """
        Decorator to register a zero‑arg *factory* under `name`.

            @AdapterRegistry.register_fn("language")
            def make_language_adapter():
                def run(model, prompt, style):
                    ...
                return run

        Returns:
            The original factory unchanged (normal Python semantics).
        """
        def _decorator(factory: Callable[[], Callable]) -> Callable[[], Callable]:
            cls.register(name, factory)  # uses normalization + locking
            return factory
        return _decorator

    # ────────────────────────────────────────────────────────────────────────
    # Optional: specs, readiness, simulate, instantiate
    # ────────────────────────────────────────────────────────────────────────

    @classmethod
    def spec(cls, name: str) -> Optional[AdapterSpec]:
        """
        Retrieve AdapterSpec attached to the factory (if any).
        """
        factory = cls.get(name)
        return getattr(factory, "__adapter_spec__", None)

    @classmethod
    def instantiate(cls, name: str) -> Callable:
        """
        Materialize a runner from the factory and attach the spec (if any)
        to the returned runner object/function for convenient introspection.
        """
        factory = cls.get(name)
        runner = factory()
        spec = getattr(factory, "__adapter_spec__", None)
        try:
            setattr(runner, "__adapter_spec__", spec)
        except Exception:
            pass
        return runner

    @classmethod
    def validate(cls, name: str, feed: Mapping[str, Any]) -> ReadyReport:
        """
        Compare a feed dict against the adapter's declared expectations.
        Legacy adapters without specs always return ok=True.
        """
        return validate_feed(feed, cls.spec(name))

    @classmethod
    def simulate(cls, name: str, **kwargs: Any) -> Any:
        """
        If the adapter exposes a simulate(**kwargs) hook (on the factory or on
        the instantiated runner), call it. Otherwise raise NotImplementedError.

        Plain words: when “wait for real data” is allowed to fall back to a
        sandbox, this is the sandbox. Adapters can choose to implement it or not.
        """
        factory = cls.get(name)
        sim = getattr(factory, "simulate", None)
        if callable(sim):
            return sim(**kwargs)

        # try the runner instance
        runner = factory()
        sim2 = getattr(runner, "simulate", None)
        if callable(sim2):
            return sim2(**kwargs)

        raise NotImplementedError(f"adapter:{_norm(name)} has no simulate()")
    

    # ────────────────────────────────────────────────────────────────────────
    # Testing / maintenance helpers (no external callers in core flow)
    # ────────────────────────────────────────────────────────────────────────

    @classmethod
    def unregister(cls, name: str) -> None:
        """Remove a factory by name. Intended for tests or dynamic reload scenarios."""
        key = _norm(name)
        with cls._lock:
            cls._reg.pop(key, None)

    @classmethod
    def clear(cls) -> None:
        """Drop all registered adapters. Intended for tests or process teardown."""
        with cls._lock:
            cls._reg.clear()
