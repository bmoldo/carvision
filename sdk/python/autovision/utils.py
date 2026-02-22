"""Preprocessing and postprocessing utilities for AutoVision."""

import json
import numpy as np
from pathlib import Path
from typing import Optional

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

INPUT_SIZE = 384


def load_image(image_path: str) -> np.ndarray:
    """Load an image as RGB numpy array.

    Supports common formats via PIL. Handles EXIF rotation automatically.
    """
    from PIL import Image, ImageOps

    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    return np.array(img)


def center_crop(image: np.ndarray) -> np.ndarray:
    """Center-crop to square using the shorter dimension."""
    h, w = image.shape[:2]
    size = min(h, w)
    top = (h - size) // 2
    left = (w - size) // 2
    return image[top : top + size, left : left + size]


def preprocess(image: np.ndarray, input_size: int = INPUT_SIZE) -> np.ndarray:
    """Full preprocessing pipeline: crop, resize, normalize.

    Returns NHWC float32 array ready for inference.
    """
    from PIL import Image

    cropped = center_crop(image)
    pil_img = Image.fromarray(cropped)
    resized = pil_img.resize((input_size, input_size), Image.BILINEAR)
    arr = np.array(resized, dtype=np.float32) / 255.0
    normalized = (arr - IMAGENET_MEAN) / IMAGENET_STD
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
