"""Exception types for the AutoVision SDK."""


class AutoVisionError(Exception):
    """Base class for all AutoVision SDK errors."""


class ModelLoadError(AutoVisionError):
    """Raised when model weights cannot be located or loaded."""


class MappingMismatchError(AutoVisionError):
    """Raised when the class mapping size does not match the manifest's num_classes."""


class InvalidImageError(AutoVisionError):
    """Raised when an image file cannot be decoded by PIL."""


class UnsupportedFormatError(AutoVisionError):
    """Raised when a model file has an unsupported format or extension."""
