# ElementFold · mesh.py
# “Resonator mesh” — a thin, fail‑safe wrapper over torch.distributed that:
#   • bootstraps a process group (NCCL on CUDA, else Gloo) ⟲ phase‑lock cluster,
#   • exposes rank/world, barrier(), shutdown(),
#   • broadcasts slow control knobs (β, γ, ⛔ clamp) from root,
#   • all‑reduces telemetry dicts with a clear op (mean/sum/min/max),
#   • degrades gracefully to single‑process when dist is unavailable.
#
# single‑GPU/CPU mode (no‑op), and becomes collective when distributed is initialized.

from __future__ import annotations
import os, contextlib
import torch

class Mesh:
    # — cluster facts (updated at init) —
    world: int = 1          # 🌍 number of processes in the group
    rank: int = 0           # 🏷  this process id   (0..world-1)
    local_rank: int = 0     # 🖥  GPU index on this node (for NCCL)
    root: int = 0           # 🎯 designated broadcaster (usually 0)
    backend: str | None = None
    ok: bool = False        # ✅ distributed is initialized and healthy
    _device: torch.device = torch.device("cpu")  # comm tensor device

    # ————————————————————————————————————————————————
    # Boot • detect env • pick backend • set device
    # ————————————————————————————————————————————————
    @classmethod
    def init(cls, backend: str | None = None, init_method: str | None = None, timeout_sec: int = 180) -> bool:
        """
        Try to initialize torch.distributed. We prefer:
          • NCCL when CUDA is available (GPU tensors required by NCCL),
          • Gloo otherwise (CPU tensors).
        We also respect standard env (RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR/PORT).
        """
        if not torch.distributed.is_available():
            cls._mark_single()
            return cls.ok

        # Pick backend automatically if not provided.
        if backend is None:
            backend = "nccl" if torch.cuda.is_available() else "gloo"

        # Respect env for ranks; default to single process if missing.
        cls.rank = int(os.environ.get("RANK", "0"))
        cls.world = int(os.environ.get("WORLD_SIZE", "1"))
        cls.local_rank = int(os.environ.get("LOCAL_RANK", str(cls.rank)))
        cls.root = 0
        cls.backend = backend

        # Choose comm tensor device: NCCL → CUDA(local_rank), Gloo → CPU
        if backend == "nccl" and torch.cuda.is_available():
            try:
                torch.cuda.set_device(cls.local_rank)
            except Exception:
                # If set_device fails, fall back to cuda:0 if present
                if torch.cuda.device_count() > 0:
                    torch.cuda.set_device(0)
                    cls.local_rank = 0
            cls._device = torch.device("cuda", cls.local_rank)
        else:
            cls._device = torch.device("cpu")

        # If already initialized, just refresh flags and return.
        if torch.distributed.is_initialized():
            cls.ok = True
            # Sync world/rank with the group in case env lied.
            try:
                cls.world = torch.distributed.get_world_size()
                cls.rank = torch.distributed.get_rank()
            except Exception:
                pass
            return cls.ok

        # Build kwargs for init_process_group; prefer env:// if addresses provided.
        kwargs = dict(backend=backend)
        if init_method is not None:
            kwargs["init_method"] = init_method
        else:
            # If MASTER_ADDR/PORT exist, env:// will work; otherwise PyTorch uses file:// fallback in some launchers.
            if "MASTER_ADDR" in os.environ and "MASTER_PORT" in os.environ:
                kwargs["init_method"] = "env://"

        # Optional timeout (older PyTorch may not accept timedelta; guard politely).
        try:
            import datetime
            kwargs["timeout"] = datetime.timedelta(seconds=int(timeout_sec))
        except Exception:
            pass

        # Try to bring the group up.
        try:
            torch.distributed.init_process_group(**kwargs)
            cls.world = torch.distributed.get_world_size()
            cls.rank = torch.distributed.get_rank()
            cls.ok = True
        except Exception:
            cls._mark_single()

        return cls.ok

    # ————————————————————————————————————————————————
    # Simple facts & hygiene
    # ————————————————————————————————————————————————
    @classmethod
    def is_primary(cls) -> bool:
        """True only on the root rank (rank 0 by default)."""
        return cls.rank == cls.root

    @classmethod
    def is_initialized(cls) -> bool:
        """Whether a process group is active and usable."""
        return bool(cls.ok and torch.distributed.is_available() and torch.distributed.is_initialized())

    @classmethod
    def barrier(cls) -> None:
        """⏸ synchronize all ranks (no‑op in single‑process)."""
        if cls.is_initialized():
            try:
                torch.distributed.barrier()
            except Exception:
                pass

    @classmethod
    def shutdown(cls) -> None:
        """🛑 tear down the process group (safe to call multiple times)."""
        if cls.is_initialized():
            try:
                torch.distributed.destroy_process_group()
            except Exception:
                pass
        cls._mark_single()

    @classmethod
    def _mark_single(cls) -> None:
        """Internal: set flags for single‑process fallback."""
        cls.world, cls.rank, cls.local_rank = 1, 0, 0
        cls.root, cls.backend, cls.ok = 0, None, False
        cls._device = torch.device("cpu")

    # ————————————————————————————————————————————————
    # Slow‑control broadcast: (β, γ, ⛔ clamp)
    # ————————————————————————————————————————————————
    @classmethod
    def broadcast_control(cls, beta: float, gamma: float, clamp: float, root: int | None = None):
        """
        Broadcast the triple (β, γ, ⛔) to all ranks. Works on CPU (Gloo) or CUDA (NCCL).
        Returns the list [β, γ, ⛔] that the *current* rank will use.
        """
        src = cls.root if root is None else int(root)
        if not cls.is_initialized():
            return [float(beta), float(gamma), float(clamp)]

        # IMPORTANT: NCCL requires CUDA tensors; Gloo accepts CPU. Use cls._device.
        x = torch.tensor([float(beta), float(gamma), float(clamp)], dtype=torch.float32, device=cls._device)
        try:
            torch.distributed.broadcast(x, src=src)
        except Exception:
            # Some backends demand matching devices across ranks; as a last resort, try CPU broadcast.
            x_cpu = x.detach().to("cpu")
            torch.distributed.broadcast(x_cpu, src=src)
            x = x_cpu
        return x.detach().cpu().tolist()

    # ————————————————————————————————————————————————
    # Telemetry reduce: dict[str → float] with clear ops
    # ————————————————————————————————————————————————
    @classmethod
    def allreduce_telemetry(cls, d: dict[str, float], op: str = "mean") -> dict[str, float]:
        """
        Reduce a small telemetry dict across ranks.
        Supported ops: 'mean', 'sum', 'min', 'max'.
        Keys must be the same on all ranks; we enforce a shared, sorted key order.

        Returns a new dict with reduced values on every rank.
        """
        if not cls.is_initialized():
            return d

        # Establish a canonical key order from root via broadcast_object_list if available.
        keys = sorted(d.keys())
        if hasattr(torch.distributed, "broadcast_object_list"):
            obj = [keys] if cls.is_primary() else [None]
            torch.distributed.broadcast_object_list(obj, src=cls.root)
            keys = obj[0]
        # Pack values into a tensor on comm device.
        vals = torch.tensor([float(d[k]) for k in keys], dtype=torch.float32, device=cls._device)

        # Map op → ReduceOp
        op_map = {
            "sum": torch.distributed.ReduceOp.SUM,
            "mean": torch.distributed.ReduceOp.SUM,  # do sum, then divide
            "min": torch.distributed.ReduceOp.MIN,
            "max": torch.distributed.ReduceOp.MAX,
        }
        if op not in op_map:
            raise ValueError(f"unsupported reduce op: {op!r}")
        torch.distributed.all_reduce(vals, op=op_map[op])

        if op == "mean":
            vals = vals / float(cls.world)

        out = {k: float(v) for k, v in zip(keys, vals.detach().cpu().tolist())}
        return out

    # ————————————————————————————————————————————————
    # Generic helpers (scalars & tensors)
    # ————————————————————————————————————————————————
    @classmethod
    def reduce_scalar(cls, x: float, op: str = "mean") -> float:
        """All‑reduce a single float across ranks."""
        if not cls.is_initialized():
            return float(x)
        t = torch.tensor([float(x)], dtype=torch.float32, device=cls._device)
        op_map = {
            "sum": torch.distributed.ReduceOp.SUM,
            "mean": torch.distributed.ReduceOp.SUM,
            "min": torch.distributed.ReduceOp.MIN,
            "max": torch.distributed.ReduceOp.MAX,
        }
        if op not in op_map:
            raise ValueError(f"unsupported reduce op: {op!r}")
        torch.distributed.all_reduce(t, op=op_map[op])
        if op == "mean":
            t /= float(cls.world)
        return float(t.item())

    @classmethod
    def broadcast_tensor(cls, t: torch.Tensor, root: int | None = None) -> torch.Tensor:
        """
        Broadcast a small tensor in‑place from root to all ranks. The tensor must be allocated on
        a device compatible with the backend (CUDA for NCCL, CPU for Gloo). We return the same tensor.
        """
        if not cls.is_initialized():
            return t
        src = cls.root if root is None else int(root)
        # Ensure device is OK; if not, move temporarily then move back.
        needs_move = (cls.backend == "nccl" and t.device.type != "cuda") or (cls.backend != "nccl" and t.device.type != "cpu")
        bak_device = t.device
        if needs_move:
            t = t.to(cls._device)
        torch.distributed.broadcast(t, src=src)
        if needs_move:
            t = t.to(bak_device)
        return t

    # ————————————————————————————————————————————————
    # Slow‑clock tick: align a global step across ranks
    # ————————————————————————————————————————————————
    @classmethod
    def tick(cls, step: int) -> int:
        """
        Keep a global “slow clock” in sync: we take the maximum step across ranks so everyone
        agrees on the most advanced tick. Useful for coarse orchestration (e.g., control sweeps).
        """
        if not cls.is_initialized():
            return int(step)
        t = torch.tensor([int(step)], dtype=torch.int64, device=cls._device)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
        return int(t.item())


# ————————————————————————————————————————————————
# Context manager sugar: with Mesh.session(): ...
# ————————————————————————————————————————————————
@contextlib.contextmanager
def session(backend: str | None = None, init_method: str | None = None):
    """
    Context that brings the mesh up and tears it down automatically.
    Usage:
        with session():
            Mesh.broadcast_control(1.0, 0.5, 5.0)
    """
    started = Mesh.init(backend=backend, init_method=init_method)
    try:
        yield started
    finally:
        # Do not shutdown if someone else initialized outside the context and expects persistence.
        # Heuristic: if WORLD_SIZE>1 in env but dist not initialized here, we still shut down because
        # most launchers isolate processes. Safe in practice.
        Mesh.shutdown()
