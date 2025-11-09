# -*- coding: utf-8 -*-
"""
Adapter capability & requirement discovery for the Studio.

Goal:
- Make every adapter visible and selectable.
- Tell users exactly what inputs/params are required and why.
- Expose a simulation toggle, but keep simulation OFF by default.
- Provide human-readable "narrative rows" per parameter.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import importlib
import pkgutil
import inspect

# ---------- Data structures ----------

@dataclass
class ParamSpec:
    key: str
    type: str
    default: Any = None
    doc: str = ""
    choices: Optional[List[Any]] = None
    required: bool = False
    state: str = "unset"  # e.g., "unset" | "provided" | "computed"
    why: str = ""         # short narrative rationale
    keybind: Optional[str] = None  # optional 1-letter/number shortcut

@dataclass
class IOSpec:
    key: str
    type: str
    doc: str = ""

@dataclass
class AdapterSpec:
    name: str
    title: str
    description: str = ""
    inputs: List[IOSpec] = field(default_factory=list)
    outputs: List[IOSpec] = field(default_factory=list)
    params: List[ParamSpec] = field(default_factory=list)
    simulate_supported: bool = False
    examples: List[str] = field(default_factory=list)
    module: Optional[str] = None  # dotted path


# ---------- Discovery ----------

def _load_module_specs(mod) -> Optional[AdapterSpec]:
    """
    Attempt to read a SPEC or get_spec() from an adapter module.
    Fallback to docstring/heuristics if absent.
    """
    # 1) Explicit SPEC object/dict
    if hasattr(mod, "SPEC"):
        spec = getattr(mod, "SPEC")
        if isinstance(spec, AdapterSpec):
            out = spec
        elif isinstance(spec, dict):
            out = AdapterSpec(
                name=spec.get("name", mod.__name__.split(".")[-1]),
                title=spec.get("title", spec.get("name", mod.__name__.split(".")[-1]).title()),
                description=spec.get("description", mod.__doc__ or ""),
                inputs=[IOSpec(**x) for x in spec.get("inputs", [])],
                outputs=[IOSpec(**x) for x in spec.get("outputs", [])],
                params=[ParamSpec(**x) for x in spec.get("params", [])],
                simulate_supported=bool(spec.get("simulate_supported", False)),
                examples=list(spec.get("examples", [])),
                module=mod.__name__,
            )
        else:
            out = None
        if out:
            if not out.module:
                out.module = mod.__name__
            return out

    # 2) Function get_spec() → dict-like
    if hasattr(mod, "get_spec") and callable(mod.get_spec):
        spec = mod.get_spec()
        if isinstance(spec, AdapterSpec):
            if not spec.module:
                spec.module = mod.__name__
            return spec
        elif isinstance(spec, dict):
            return AdapterSpec(
                name=spec.get("name", mod.__name__.split(".")[-1]),
                title=spec.get("title", spec.get("name", mod.__name__.split(".")[-1]).title()),
                description=spec.get("description", mod.__doc__ or ""),
                inputs=[IOSpec(**x) for x in spec.get("inputs", [])],
                outputs=[IOSpec(**x) for x in spec.get("outputs", [])],
                params=[ParamSpec(**x) for x in spec.get("params", [])],
                simulate_supported=bool(spec.get("simulate_supported", False)),
                examples=list(spec.get("examples", [])),
                module=mod.__name__,
            )

    # 3) Heuristic fallback
    sim_supported = hasattr(mod, "simulate") or any(name.startswith("simulate_") for name, _ in inspect.getmembers(mod, inspect.isfunction))
    doc = (mod.__doc__ or "").strip()
    return AdapterSpec(
        name=mod.__name__.split(".")[-1],
        title=mod.__name__.split(".")[-1].replace("_"," ").title(),
        description=doc,
        inputs=[],
        outputs=[],
        params=[],
        simulate_supported=sim_supported,
        examples=[],
        module=mod.__name__,
    )


def discover_adapter_specs(package: str = "elementfold.experience.adapters") -> List[AdapterSpec]:
    """Walk the adapters package and build specs for each module."""
    specs: List[AdapterSpec] = []
    pkg = importlib.import_module(package)
    for m in pkgutil.iter_modules(pkg.__path__, pkg.__name__ + "."):
        name = m.name.split(".")[-1]
        # Skip non-adapter infra modules by convention
        if name in {"base", "__init__"}:
            continue
        mod = importlib.import_module(m.name)
        spec = _load_module_specs(mod)
        if spec:
            specs.append(spec)
    # Stable order by title
    specs.sort(key=lambda s: s.title.lower())
    return specs


# ---------- Narrative rows ----------

def narrative_rows_for_spec(spec: AdapterSpec) -> List[str]:
    """
    Produce a human-readable list of rows describing params and IO.
    'A char used as a parameter has its own row': we map keybind if provided.
    """
    rows: List[str] = []
    rows.append(f"[{spec.title}] — {spec.description}".strip())
    if spec.inputs:
        rows.append("Inputs:")
        for io in spec.inputs:
            rows.append(f"  • {io.key} : {io.type} — {io.doc}")
    if spec.outputs:
        rows.append("Outputs:")
        for io in spec.outputs:
            rows.append(f"  • {io.key} : {io.type} — {io.doc}")
    if spec.params:
        rows.append("Parameters:")
        for p in spec.params:
            kb = f"[{p.keybind}] " if p.keybind else ""
            req = "required" if p.required else "optional"
            state = f"state={p.state}"
            default = "" if p.default is None else f"default={p.default!r}; "
            rows.append(f"  • {kb}{p.key} ({p.type}, {req}) — {default}{state}. {p.doc} {p.why}".strip())
    rows.append(f"Simulation supported: {'yes' if spec.simulate_supported else 'no'}")
    if spec.examples:
        rows.append("Examples:")
        for e in spec.examples:
            rows.append(f"  • {e}")
    return rows


# ---------- Simple menu model ----------

def menu_items_for_specs(specs: List[AdapterSpec]) -> List[str]:
    """Return menu lines: one adapter per row."""
    lines = []
    for i, s in enumerate(specs, 1):
        sim = " [sim]" if s.simulate_supported else ""
        lines.append(f"{i}. {s.title}{sim}  —  {s.description[:80]}")
    return lines


# ---------- Public API for Studio ----------

def list_adapters() -> List[Dict[str, Any]]:
    """Return a simple list for UI layers."""
    out = []
    for s in discover_adapter_specs():
        out.append({
            "name": s.name,
            "title": s.title,
            "module": s.module,
            "simulate_supported": s.simulate_supported,
        })
    return out


def get_adapter_spec(name: str) -> AdapterSpec:
    for s in discover_adapter_specs():
        if s.name == name or s.title.lower() == name.lower():
            return s
    raise KeyError(f"Adapter not found: {name}")


def pretty_print_spec(name: str) -> str:
    s = get_adapter_spec(name)
    return "\n".join(narrative_rows_for_spec(s))


__all__ = [
    "ParamSpec","IOSpec","AdapterSpec",
    "discover_adapter_specs","narrative_rows_for_spec",
    "menu_items_for_specs","list_adapters","get_adapter_spec","pretty_print_spec",
]
