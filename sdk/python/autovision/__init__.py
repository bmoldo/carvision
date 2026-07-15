from .classifier import AutoVision, ClassificationResult, Prediction
from .errors import (
    AutoVisionError,
    InvalidImageError,
    MappingMismatchError,
    ModelLoadError,
    UnsupportedFormatError,
)

__version__ = "0.2.0"
__all__ = [
    "AutoVision",
    "ClassificationResult",
    "Prediction",
    "AutoVisionError",
    "ModelLoadError",
    "MappingMismatchError",
    "InvalidImageError",
    "UnsupportedFormatError",
]
