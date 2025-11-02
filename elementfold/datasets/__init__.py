# ElementFold · datasets/__init__.py
# One place to import text / image / audio loaders and the synthetic click‑pulse generator.

from .text_files import (
    iter_text_paths,
    TextLineDataset,
    TextChunkDataset,
    pad_collate,
    make_text_loader,
    load_folder as load_text_folder,
)

from .image_folder import (
    iter_image_paths,
    ImageFolderDataset,
    make_image_loader as make_image_loader,
    load_folder as load_image_folder,
)

from .audio_folder import (
    AudioFolderDataset,
)

from .grid_pulse_synth import (
    DEFAULT_DELTA,
    DEFAULT_CAPS,
    make_pulse,
    ClickPulseDataset,
    make_pulse_loader,
)

__all__ = [
    # text
    "iter_text_paths",
    "TextLineDataset",
    "TextChunkDataset",
    "pad_collate",
    "make_text_loader",
    "load_text_folder",
    # images
    "iter_image_paths",
    "ImageFolderDataset",
    "make_image_loader",
    "load_image_folder",
    # audio
    "AudioFolderDataset",
    # synthetic click‑pulse
    "DEFAULT_DELTA",
    "DEFAULT_CAPS",
    "make_pulse",
    "ClickPulseDataset",
    "make_pulse_loader",
]
