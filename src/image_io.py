"""Shared image range and dtype handling for training and inference."""

from __future__ import annotations

import numpy as np


def normalize_image_array(array: np.ndarray) -> np.ndarray:
    """Convert an image to float32 using its storage dtype's full range.

    Floating-point arrays are returned unchanged because NPY inputs may
    intentionally contain out-of-range degraded values. Integer image files
    are scaled from their native dtype, not from their observed maximum.
    """
    source_dtype = array.dtype
    result = array.astype(np.float32)
    if np.issubdtype(source_dtype, np.unsignedinteger):
        return result / float(np.iinfo(source_dtype).max)
    return result


def image_storage_dtype(path) -> np.dtype:
    """Return the stored dtype of an NPY or Pillow-readable image."""
    from pathlib import Path

    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path, mmap_mode="r").dtype
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image).dtype
