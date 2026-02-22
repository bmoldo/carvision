"""AutoVision — car make/model/generation classifier."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .utils import load_image, preprocess, softmax, load_class_mapping


@dataclass
class Prediction:
    """A single classification prediction."""

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


class AutoVision:
    """Car recognition engine.

    Supports TFLite and ONNX backends. Automatically selects based on model
    file extension.

    Args:
        model_path: Path to .tflite or .onnx model file.
        class_mapping_path: Path to class_mapping.json. If None, looks for it
            next to the model file.
        temperature: Softmax temperature for calibrated probabilities.
    """

    def __init__(
        self,
        model_path: str,
        class_mapping_path: Optional[str] = None,
        temperature: float = 1.0,
    ):
        self.model_path = Path(model_path)
        self.temperature = temperature

        # Load class mapping
        if class_mapping_path is None:
            mapping_path = self.model_path.parent / "class_mapping.json"
        else:
            mapping_path = Path(class_mapping_path)
        self.class_mapping = load_class_mapping(str(mapping_path))

        # Load model based on extension
        ext = self.model_path.suffix.lower()
        if ext == ".tflite":
            self._backend = "tflite"
            self._load_tflite()
        elif ext == ".onnx":
            self._backend = "onnx"
            self._load_onnx()
        else:
            raise ValueError(f"Unsupported model format: {ext}. Use .tflite or .onnx")

    def _load_tflite(self) -> None:
        try:
            import tflite_runtime.interpreter as tflite

            self._interpreter = tflite.Interpreter(model_path=str(self.model_path))
        except ImportError:
            import tensorflow as tf

            self._interpreter = tf.lite.Interpreter(model_path=str(self.model_path))

        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

    def _load_onnx(self) -> None:
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            str(self.model_path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

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

    def classify(
        self,
        image_path: str,
        top_k: int = 5,
    ) -> list[Prediction]:
        """Classify a car image.

        Args:
            image_path: Path to an image file (JPEG, PNG, etc.).
            top_k: Number of top predictions to return.

        Returns:
            List of Prediction objects sorted by confidence (descending).
        """
        image = load_image(image_path)
        input_tensor = preprocess(image)
        logits = self._infer(input_tensor)
        probabilities = softmax(logits, temperature=self.temperature)

        top_indices = np.argsort(probabilities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            idx = int(idx)
            cls = self.class_mapping[idx]
            results.append(
                Prediction(
                    index=idx,
                    class_name=cls["class_name"],
                    make=cls["make"],
                    model=cls["model"],
                    generation=cls.get("generation"),
                    year_start=cls.get("year_start"),
                    year_end=cls.get("year_end"),
                    rarity=cls.get("rarity", "UNKNOWN"),
                    confidence=float(probabilities[idx]),
                    logit=float(logits[idx]),
                )
            )

        return results

    def classify_batch(
        self,
        image_paths: list[str],
        top_k: int = 5,
    ) -> list[list[Prediction]]:
        """Classify multiple images.

        Args:
            image_paths: List of image file paths.
            top_k: Number of top predictions per image.

        Returns:
            List of prediction lists, one per image.
        """
        return [self.classify(path, top_k=top_k) for path in image_paths]
