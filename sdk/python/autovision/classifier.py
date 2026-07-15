"""AutoVision — car make/model/generation classifier."""

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .errors import (
    MappingMismatchError,
    ModelLoadError,
    UnsupportedFormatError,
)
from .utils import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    find_file,
    load_class_mapping,
    load_confusion_pairs,
    load_image,
    load_manifest,
    preprocess,
    softmax,
)

ENGINE_VERSION = "0.2.0"

BACKGROUND_RARITY = "BACKGROUND"

# Fallback confidence threshold for rarities not present in the manifest.
DEFAULT_RARITY_THRESHOLD = 0.5

# Built-in defaults matching the v5.13.0 release manifest. Used only when no
# model_manifest.json can be found next to a legacy model path.
DEFAULT_MANIFEST: dict = {
    "version": "5.13.0",
    "num_classes": 897,
    "input_size": 384,
    "normalization": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "temperature": 1.0,
    "quantization": "float16",
    "ood_threshold": 0.05,
    "confidence_thresholds": {
        "COMMON": 0.4,
        "UNCOMMON": 0.5,
        "RARE": 0.6,
        "ULTRA_RARE": 0.65,
        "EPIC": 0.67,
        "LEGENDARY": 0.7,
    },
    "confusion_pair_margin": 0.08,
}

# Weight file names probed inside a release directory, in priority order.
_WEIGHT_CANDIDATES = ["car_classifier.tflite", "model.tflite"]


@dataclass
class Prediction:
    """A single classification prediction."""

    rank: int
    index: int
    class_name: str
    make: str
    model: str
    generation: Optional[str]
    year_start: Optional[int]
    year_end: Optional[int]
    rarity: str
    confidence: float
    logit: float


@dataclass
class ClassificationResult:
    """Full result of classifying one image.

    Attributes:
        predictions: Top-k car predictions (background class excluded),
            sorted by confidence descending, populated even when rejected.
        top1: The accepted top prediction, or None when rejected.
        rejected: True when the image failed one of the gating checks.
        rejection_reason: One of "not_a_car", "low_confidence", "ambiguous",
            or None when accepted.
        inference_ms: Model inference latency in milliseconds.
        engine_version: SDK version that produced this result.
        model_version: Model release version from the manifest.
        taxonomy_version: "{model_version}-{num_classes}".
    """

    predictions: list[Prediction] = field(default_factory=list)
    top1: Optional[Prediction] = None
    rejected: bool = False
    rejection_reason: Optional[str] = None
    inference_ms: float = 0.0
    engine_version: str = ENGINE_VERSION
    model_version: str = ""
    taxonomy_version: str = ""


class AutoVision:
    """Car recognition engine.

    Supports TFLite and ONNX backends, selected by model file extension.
    The model manifest (model_manifest.json) is the source of truth for
    input size, normalization, temperature and confidence gating thresholds.

    Args:
        model_path: Either a release directory containing model_manifest.json,
            class_mapping.json and model weights (e.g. "models/v5.13.0"), or a
            direct path to a .tflite/.onnx model file.
        class_mapping_path: Path to class_mapping.json. If None, looks next to
            the model file.
        manifest_path: Path to model_manifest.json. If None, looks next to the
            model file; falls back to built-in v5.13.0 defaults with a warning.
        temperature: Optional softmax temperature override. If None, the
            manifest value is used.

    Raises:
        ModelLoadError: If model weights cannot be found or loaded.
        MappingMismatchError: If class mapping size != manifest num_classes.
        UnsupportedFormatError: If the model file extension is unsupported.
    """

    def __init__(
        self,
        model_path: str,
        class_mapping_path: Optional[str] = None,
        manifest_path: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        path = Path(model_path)

        if path.is_dir():
            release_dir = path
            self.model_path = self._find_weights(release_dir)
        else:
            release_dir = path.parent
            self.model_path = path

        # --- Manifest (source of truth for preprocessing + gating) ---
        if manifest_path is None:
            found = find_file(release_dir, ["model_manifest.json"])
            if found is not None:
                self.manifest = load_manifest(str(found))
            else:
                warnings.warn(
                    f"No model_manifest.json found in {release_dir}; "
                    "falling back to built-in v5.13.0 defaults.",
                    stacklevel=2,
                )
                self.manifest = dict(DEFAULT_MANIFEST)
        else:
            self.manifest = load_manifest(manifest_path)

        self.model_version: str = str(self.manifest.get("version", "unknown"))
        self.engine_version: str = ENGINE_VERSION
        self.num_classes: int = int(
            self.manifest.get("num_classes", DEFAULT_MANIFEST["num_classes"])
        )
        self.taxonomy_version: str = f"{self.model_version}-{self.num_classes}"
        self.input_size: int = int(
            self.manifest.get("input_size", DEFAULT_MANIFEST["input_size"])
        )
        self.quantization: str = str(self.manifest.get("quantization", "unknown"))

        normalization = self.manifest.get("normalization", {})
        self.mean = np.array(
            normalization.get("mean", IMAGENET_MEAN), dtype=np.float32
        )
        self.std = np.array(
            normalization.get("std", IMAGENET_STD), dtype=np.float32
        )

        if temperature is not None:
            self.temperature = float(temperature)
        else:
            self.temperature = float(self.manifest.get("temperature", 1.0))

        self.ood_threshold: float = float(
            self.manifest.get("ood_threshold", DEFAULT_MANIFEST["ood_threshold"])
        )
        self.confidence_thresholds: dict = dict(
            self.manifest.get(
                "confidence_thresholds", DEFAULT_MANIFEST["confidence_thresholds"]
            )
        )
        self.confusion_pair_margin: float = float(
            self.manifest.get(
                "confusion_pair_margin", DEFAULT_MANIFEST["confusion_pair_margin"]
            )
        )

        # --- Class mapping ---
        if class_mapping_path is None:
            mapping_path = release_dir / "class_mapping.json"
        else:
            mapping_path = Path(class_mapping_path)
        self.class_mapping = load_class_mapping(str(mapping_path))

        if len(self.class_mapping) != self.num_classes:
            raise MappingMismatchError(
                f"class_mapping has {len(self.class_mapping)} entries but the "
                f"manifest declares num_classes={self.num_classes}"
            )

        self.num_makes: int = len(
            {
                cls["make"]
                for cls in self.class_mapping
                if cls.get("rarity") != BACKGROUND_RARITY
            }
        )

        # --- Confusion pairs (optional) ---
        # Maps frozenset({class_a, class_b}) -> margin_threshold.
        self._confusion_pairs: dict = {}
        pairs_file = find_file(release_dir, ["confusion_pairs.json"])
        if pairs_file is None and class_mapping_path is not None:
            pairs_file = find_file(mapping_path.parent, ["confusion_pairs.json"])
        if pairs_file is not None:
            for pair in load_confusion_pairs(str(pairs_file)):
                key = frozenset((pair["class_a"], pair["class_b"]))
                self._confusion_pairs[key] = pair.get("margin_threshold")

        # --- Model backend ---
        ext = self.model_path.suffix.lower()
        if ext == ".tflite":
            self._backend = "tflite"
            self._load_tflite()
        elif ext == ".onnx":
            self._backend = "onnx"
            self._load_onnx()
        else:
            raise UnsupportedFormatError(
                f"Unsupported model format: {ext}. Use .tflite or .onnx"
            )

    @property
    def backend(self) -> str:
        """Inference backend in use ("tflite" or "onnx")."""
        return self._backend

    @staticmethod
    def _find_weights(directory: Path) -> Path:
        """Auto-detect model weights inside a release directory."""
        found = find_file(directory, _WEIGHT_CANDIDATES)
        if found is not None:
            return found
        onnx_files = sorted(directory.glob("*.onnx"))
        if onnx_files:
            return onnx_files[0]
        raise ModelLoadError(
            f"No model weights found in {directory} "
            f"(looked for {', '.join(_WEIGHT_CANDIDATES)} and *.onnx)"
        )

    def _load_tflite(self) -> None:
        if not self.model_path.is_file():
            raise ModelLoadError(f"Model file not found: {self.model_path}")
        try:
            try:
                import tflite_runtime.interpreter as tflite

                self._interpreter = tflite.Interpreter(
                    model_path=str(self.model_path)
                )
            except ImportError:
                import tensorflow as tf

                self._interpreter = tf.lite.Interpreter(
                    model_path=str(self.model_path)
                )

            self._interpreter.allocate_tensors()
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()
        except ImportError as exc:
            raise ModelLoadError(
                "TFLite backend requires tflite-runtime or tensorflow"
            ) from exc
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load TFLite model: {self.model_path}"
            ) from exc

    def _load_onnx(self) -> None:
        if not self.model_path.is_file():
            raise ModelLoadError(f"Model file not found: {self.model_path}")
        try:
            import onnxruntime as ort

            self._session = ort.InferenceSession(
                str(self.model_path),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self._input_name = self._session.get_inputs()[0].name
        except ImportError as exc:
            raise ModelLoadError("ONNX backend requires onnxruntime") from exc
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load ONNX model: {self.model_path}"
            ) from exc

    def _infer(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run inference and return raw logits."""
        if self._backend == "tflite":
            self._interpreter.set_tensor(
                self._input_details[0]["index"], input_tensor
            )
            self._interpreter.invoke()
            output = self._interpreter.get_tensor(self._output_details[0]["index"])
            return output[0]
        else:
            outputs = self._session.run(None, {self._input_name: input_tensor})
            return outputs[0][0]

    def _make_prediction(
        self, rank: int, index: int, probabilities: np.ndarray, logits: np.ndarray
    ) -> Prediction:
        cls = self.class_mapping[index]
        return Prediction(
            rank=rank,
            index=index,
            class_name=cls["class_name"],
            make=cls["make"],
            model=cls["model"],
            generation=cls.get("generation"),
            year_start=cls.get("year_start"),
            year_end=cls.get("year_end"),
            rarity=cls.get("rarity", "UNKNOWN"),
            confidence=float(probabilities[index]),
            logit=float(logits[index]),
        )

    def _build_result(
        self,
        logits: np.ndarray,
        inference_ms: float,
        top_k: int,
    ) -> ClassificationResult:
        """Apply softmax, gating logic and top-k selection to raw logits.

        Pure postprocessing — does not touch the model backend, so it can be
        exercised with synthetic logits.
        """
        probabilities = softmax(logits, temperature=self.temperature)
        order = np.argsort(probabilities)[::-1]

        # Ordered class indices excluding the background class.
        car_order = [
            int(i)
            for i in order
            if self.class_mapping[int(i)].get("rarity") != BACKGROUND_RARITY
        ]

        predictions = [
            self._make_prediction(rank, idx, probabilities, logits)
            for rank, idx in enumerate(car_order[:top_k], start=1)
        ]

        # --- Gating: applied to the argmax over ALL classes ---
        top_idx = int(order[0])
        top_prob = float(probabilities[top_idx])
        top_rarity = self.class_mapping[top_idx].get("rarity", "UNKNOWN")

        rejected = False
        rejection_reason: Optional[str] = None

        if top_rarity == BACKGROUND_RARITY:
            rejected, rejection_reason = True, "not_a_car"
        elif top_prob < self.ood_threshold:
            rejected, rejection_reason = True, "not_a_car"
        elif top_prob < self.confidence_thresholds.get(
            top_rarity, DEFAULT_RARITY_THRESHOLD
        ):
            rejected, rejection_reason = True, "low_confidence"
        elif len(car_order) >= 2:
            name_a = self.class_mapping[car_order[0]]["class_name"]
            name_b = self.class_mapping[car_order[1]]["class_name"]
            pair_key = frozenset((name_a, name_b))
            if pair_key in self._confusion_pairs:
                margin = self._confusion_pairs[pair_key]
                if margin is None:
                    margin = self.confusion_pair_margin
                p1 = float(probabilities[car_order[0]])
                p2 = float(probabilities[car_order[1]])
                if (p1 - p2) < margin:
                    rejected, rejection_reason = True, "ambiguous"

        top1 = None if rejected else (predictions[0] if predictions else None)

        return ClassificationResult(
            predictions=predictions,
            top1=top1,
            rejected=rejected,
            rejection_reason=rejection_reason,
            inference_ms=inference_ms,
            engine_version=self.engine_version,
            model_version=self.model_version,
            taxonomy_version=self.taxonomy_version,
        )

    def classify(
        self,
        image_path: str,
        top_k: int = 5,
    ) -> ClassificationResult:
        """Classify a car image.

        Args:
            image_path: Path to an image file (JPEG, PNG, etc.).
            top_k: Number of top predictions to include in the result.

        Returns:
            ClassificationResult with top-k predictions (background class
            excluded), gating outcome and version metadata.

        Raises:
            InvalidImageError: If the image cannot be decoded.
        """
        image = load_image(image_path)
        input_tensor = preprocess(
            image, input_size=self.input_size, mean=self.mean, std=self.std
        )

        start = time.perf_counter()
        logits = np.asarray(self._infer(input_tensor), dtype=np.float32)
        inference_ms = (time.perf_counter() - start) * 1000

        return self._build_result(logits, inference_ms, top_k)

    def classify_batch(
        self,
        image_paths: list[str],
        top_k: int = 5,
    ) -> list[ClassificationResult]:
        """Classify multiple images.

        Args:
            image_paths: List of image file paths.
            top_k: Number of top predictions per image.

        Returns:
            List of ClassificationResult objects, one per image.
        """
        return [self.classify(path, top_k=top_k) for path in image_paths]
