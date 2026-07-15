"""Preprocessing and postprocessing utilities for AutoVision."""

import json
import numpy as np
from pathlib import Path
from typing import Optional

from .errors import InvalidImageError

# ImageNet normalization constants (defaults; the model manifest is the
# source of truth when available).
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

INPUT_SIZE = 384


def load_image(image_path: str) -> np.ndarray:
    """Load an image as RGB numpy array.

    Supports common formats via PIL. Handles EXIF rotation automatically.

    Raises:
        FileNotFoundError: If the file does not exist.
        InvalidImageError: If PIL cannot decode the file.
    """
    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
    except FileNotFoundError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(f"Cannot decode image: {image_path}") from exc
    return np.array(img)


def center_crop(image: np.ndarray) -> np.ndarray:
    """Center-crop to square using the shorter dimension."""
    h, w = image.shape[:2]
    size = min(h, w)
    top = (h - size) // 2
    left = (w - size) // 2
    return image[top : top + size, left : left + size]


def preprocess(
    image: np.ndarray,
    input_size: int = INPUT_SIZE,
    mean: np.ndarray = IMAGENET_MEAN,
    std: np.ndarray = IMAGENET_STD,
) -> np.ndarray:
    """Full preprocessing pipeline: crop, resize, normalize.

    Returns NHWC float32 array ready for inference.
    """
    from PIL import Image

    cropped = center_crop(image)
    pil_img = Image.fromarray(cropped)
    resized = pil_img.resize((input_size, input_size), Image.BILINEAR)
    arr = np.array(resized, dtype=np.float32) / 255.0
    normalized = (arr - mean) / std
    return np.expand_dims(normalized, axis=0)  # [1, H, W, 3]


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Temperature-scaled softmax."""
    scaled = logits / temperature
    shifted = scaled - np.max(scaled)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def load_class_mapping(path: str) -> list[dict]:
    """Load class_mapping.json and return list of class metadata dicts."""
    with open(path, "r") as f:
        return json.load(f)


def load_manifest(path: str) -> dict:
    """Load model_manifest.json and return the manifest dict."""
    with open(path, "r") as f:
        return json.load(f)


def load_confusion_pairs(path: str) -> list[dict]:
    """Load confusion_pairs.json and return a list of pair dicts.

    Each entry has at least ``class_a``, ``class_b`` and ``margin_threshold``.
    """
    with open(path, "r") as f:
        return json.load(f)


def find_file(directory: Path, names: list[str]) -> Optional[Path]:
    """Return the first existing file in ``directory`` matching ``names``."""
    for name in names:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None
