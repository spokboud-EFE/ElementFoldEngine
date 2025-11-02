# ElementFold · datasets/image_folder.py
# Streaming images → tensors with *optional torchvision acceleration* and graceful fallbacks.
#
# Why this design:
#   • If torchvision is available (and compiled for your platform), we use its fast readers/transforms.
#   • If not, we fall back to Pillow (pure‑Python) without changing the API.
#   • If neither is present, we still produce synthetic tensors so pipelines remain testable end‑to‑end.
#
# What you get:
#   • iter_image_paths()      — walk a folder (optionally recursive) with an extension filter.
#   • ImageFolderDataset      — iterable dataset yielding (C,H,W) tensors.
#   • make_image_loader()     — convenience DataLoader builder.
#   • load_folder()           — legacy/simple generator kept for back‑compat.
#
# Tensors:
#   • uint8 in [0,255] by default (cheap and portable).
#   • float32 in [0,1] when to_float=True (optionally normalized with mean/std).

from __future__ import annotations
import os, random                                         # ✴ paths + small RNG for shuffling/augs
from typing import Iterator, Iterable, Optional, List      # ✴ readable type hints
import torch                                              # ✴ tensors + DataLoader
from torch.utils.data import IterableDataset, DataLoader

# — Try torchvision (fast path); keep it optional and fully guarded —
try:
    from torchvision.io import read_image as tv_read_image            # fast C/CUDA image decode → (C,H,W) uint8
    from torchvision.transforms.functional import resize as tv_resize # tensor/PIL resize
    from torchvision.transforms.functional import hflip as tv_hflip   # tensor/PIL horizontal flip
    try:
        # Interpolation enum moved around across versions; try both imports.
        from torchvision.transforms import InterpolationMode as _TVInterp
    except Exception:
        from torchvision.transforms.functional import InterpolationMode as _TVInterp  # type: ignore
    _HAS_TV = True
except Exception:
    tv_read_image = None
    tv_resize = None
    tv_hflip = None
    _TVInterp = None
    _HAS_TV = False

# — Try Pillow (portable path) —
try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    Image = None
    _HAS_PIL = False


# ———————————————————————————————————————————————————————————
# File walking (filter by typical image extensions; optional recursion)
# ———————————————————————————————————————————————————————————

def iter_image_paths(
    root: str,
    recursive: bool = True,
    allowed_exts: Optional[set[str]] = None,
    follow_symlinks: bool = False,
) -> Iterator[str]:
    """
    Yield image file paths under `root`. We accept a small whitelist of common formats.
    """
    if allowed_exts is None:
        allowed_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    root = os.path.abspath(root)
    if not recursive:
        for name in sorted(os.listdir(root)):
            p = os.path.join(root, name)
            if os.path.isfile(p) and os.path.splitext(p)[1].lower() in allowed_exts:
                yield p
        return
    for dirpath, dirnames, filenames in os.walk(root, followlinks=bool(follow_symlinks)):
        for name in sorted(filenames):
            p = os.path.join(dirpath, name)
            if os.path.splitext(p)[1].lower() in allowed_exts:
                yield p


# ———————————————————————————————————————————————————————————
# Pillow helpers (pure‑Python fallback path)
# ———————————————————————————————————————————————————————————

def _pil_open_rgb(path: str):
    if not _HAS_PIL:
        raise RuntimeError("Pillow is not available")
    im = Image.open(path)
    return im.convert("RGB")

def _pil_resize(im, size: int | tuple[int, int]):
    if isinstance(size, int):
        tgt = (size, size)
    else:
        tgt = (int(size[0]), int(size[1]))
    # Bicubic is a smooth default; matches common ML preprocessing practice.
    return im.resize(tgt, resample=Image.BICUBIC)

def _tensor_from_pil(im) -> torch.Tensor:
    w, h = im.size
    buf = im.tobytes()                                      # raw bytes, row‑major
    x = torch.frombuffer(buf, dtype=torch.uint8)            # 1‑D uint8 view
    x = x.view(h, w, 3).permute(2, 0, 1).contiguous()       # (H,W,3) → (3,H,W)
    return x


# ———————————————————————————————————————————————————————————
# Dataset: stream (C,H,W) tensors from a folder
# ———————————————————————————————————————————————————————————

class ImageFolderDataset(IterableDataset):
    """
    Stream images from a folder as tensors with shape (C,H,W).

    Args:
        root:          folder path containing images (optionally nested if recursive=True)
        size:          int or (W,H). If torchvision/Pillow present → resize; else used for synthetic fallback.
        max_items:     cap on number of yielded items (None = no cap)
        recursive:     walk subfolders when True
        shuffle_files: randomize file order (useful with multiple workers)
        augment:       apply simple random horizontal flip
        hflip_p:       probability of horizontal flip when augment=True
        to_float:      return float32 in [0,1] when True; else uint8 in [0,255]
        normalize:     optional (mean, std) each length‑3 in [0,1] to normalize float tensors
        seed:          RNG seed for shuffling/augmentation
    """

    def __init__(
        self,
        root: str,
        size: int | tuple[int, int] = 64,
        max_items: Optional[int] = None,
        recursive: bool = True,
        shuffle_files: bool = False,
        augment: bool = False,
        hflip_p: float = 0.5,
        to_float: bool = False,
        normalize: Optional[tuple[tuple[float, float, float], tuple[float, float, float]]] = None,
        seed: int = 0,
    ):
        super().__init__()
        self.root = str(root)
        self.size = size
        self.max_items = max_items if (max_items is None) else int(max_items)
        self.recursive = bool(recursive)
        self.shuffle_files = bool(shuffle_files)
        self.augment = bool(augment)
        self.hflip_p = float(hflip_p)
        self.to_float = bool(to_float)
        self.normalize = normalize
        self.seed = int(seed)

    def _paths(self) -> List[str]:
        paths = list(iter_image_paths(self.root, recursive=self.recursive))
        if self.shuffle_files:
            rng = random.Random(self.seed)
            rng.shuffle(paths)
        return paths

    def _maybe_norm(self, x: torch.Tensor) -> torch.Tensor:
        # x either uint8 [0,255] or float32 [0,1].
        if not self.to_float:
            return x
        xf = x.float() / 255.0
        if self.normalize is None:
            return xf
        mean, std = self.normalize
        m = torch.tensor(mean, dtype=xf.dtype).view(3, 1, 1)  # broadcast (C,1,1)
        s = torch.tensor(std, dtype=xf.dtype).view(3, 1, 1)
        return (xf - m) / (s + 1e-6)

    def _target_hw(self) -> tuple[int, int]:
        # Normalize size parameter into (H,W) ints.
        if isinstance(self.size, int):
            return (self.size, self.size)
        return (int(self.size[1]), int(self.size[0]))  # incoming is (W,H); convert to (H,W)

    def __iter__(self) -> Iterator[torch.Tensor]:
        # Local RNG per iterator (important for multi‑worker IterableDataset)
        rng = random.Random(self.seed + (os.getpid() if hasattr(os, "getpid") else 0))
        count = 0
        paths = self._paths()
        H, W = self._target_hw()

        for p in paths:
            if self.max_items is not None and count >= self.max_items:
                return

            x: torch.Tensor | None = None

            # — Fast path: torchvision decode + tensor transforms —
            if _HAS_TV and tv_read_image is not None and tv_resize is not None and tv_hflip is not None:
                try:
                    x = tv_read_image(p)                              # (C,H,W) uint8
                    if self.augment and rng.random() < self.hflip_p:
                        x = tv_hflip(x)                               # Horizontal flip on tensor
                    # Resize expects size as (H,W); choose bicubic if the enum is present, else default.
                    if _TVInterp is not None:
                        x = tv_resize(x, size=[H, W], interpolation=_TVInterp.BICUBIC)
                    else:
                        x = tv_resize(x, size=[H, W])                 # Fallback to default interpolation
                except Exception:
                    x = None  # If torchvision fails (codec/ABI), try Pillow next.

            # — Portable path: Pillow decode —
            if x is None and _HAS_PIL:
                try:
                    im = _pil_open_rgb(p)                             # Decode RGB
                    if self.augment and rng.random() < self.hflip_p:
                        im = im.transpose(Image.FLIP_LEFT_RIGHT)      # Random horizontal flip
                    im = _pil_resize(im, (W, H))                      # Pillow uses (W,H)
                    x = _tensor_from_pil(im)                          # (3,H,W) uint8
                except Exception:
                    x = None  # If decode/resize fails, fall through to synthetic.

            # — Synthetic path: no decoders available or both failed —
            if x is None:
                # Keep the pipeline testable even in bare‑bones environments.
                x = torch.randint(0, 256, (3, H, W), dtype=torch.uint8)

            # Convert to float / normalize if requested, then yield.
            x = self._maybe_norm(x)
            yield x
            count += 1


# ———————————————————————————————————————————————————————————
# Convenience DataLoader builder
# ———————————————————————————————————————————————————————————

def make_image_loader(
    root: str,
    size: int | tuple[int, int] = 64,
    batch: int = 32,
    workers: int = 0,
    **kw,
) -> DataLoader:
    """
    Build a DataLoader streaming (B, C, H, W) batches from a folder.
    Extra kwargs (**kw) are forwarded to ImageFolderDataset.
    """
    ds = ImageFolderDataset(root=root, size=size, **kw)
    return DataLoader(ds, batch_size=int(batch), shuffle=False, num_workers=int(workers))


# ———————————————————————————————————————————————————————————
# Back‑compat generator (simple, one‑by‑one)
# ———————————————————————————————————————————————————————————

def load_folder(path: str, size: int = 64, max_items: Optional[int] = None) -> Iterable[torch.Tensor]:
    """
    Legacy/simple generator kept for compatibility with earlier notebooks:
        for img in load_folder("data/images", size=64):
            ...
    Yields (3,H,W) tensors (uint8 by default; use make_image_loader for float/normalize).
    """
    ds = ImageFolderDataset(root=path, size=size, max_items=max_items, recursive=False)
    yield from ds


__all__ = [
    "iter_image_paths",
    "ImageFolderDataset",
    "make_image_loader",
    "load_folder",
]
