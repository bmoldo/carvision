"""AutoVision API server — car recognition via HTTP."""

import os
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

# Add SDK to path
sys.path.insert(0, str(Path(__file__).parent.parent / "sdk" / "python"))

from autovision import AutoVision, ClassificationResult, InvalidImageError, Prediction
from autovision.classifier import ENGINE_VERSION

# Model location. AUTOVISION_MODEL_DIR (release directory) is preferred;
# AUTOVISION_MODEL / AUTOVISION_CLASSES are honored for backwards compatibility.
MODEL_DIR = os.environ.get("AUTOVISION_MODEL_DIR", "/app/models/v5.13.0")
MODEL_PATH = os.environ.get("AUTOVISION_MODEL")
CLASS_MAPPING_PATH = os.environ.get("AUTOVISION_CLASSES")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_TOP_K = 20
ACCEPTED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

engine: Optional[AutoVision] = None


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    """Build a structured error response: {"error": {"code", "message"}}."""
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


def _load_engine() -> None:
    global engine
    try:
        if MODEL_PATH:
            engine = AutoVision(
                model_path=MODEL_PATH,
                class_mapping_path=(
                    CLASS_MAPPING_PATH
                    if CLASS_MAPPING_PATH and Path(CLASS_MAPPING_PATH).exists()
                    else None
                ),
            )
            print(f"Model loaded from {MODEL_PATH}")
        elif Path(MODEL_DIR).is_dir():
            engine = AutoVision(MODEL_DIR)
            print(f"Model loaded from {MODEL_DIR}")
        else:
            print(
                f"WARNING: Model directory not found at {MODEL_DIR}. "
                "/classify will return 503."
            )
    except Exception as exc:
        engine = None
        print(f"WARNING: Failed to load model ({type(exc).__name__}). "
              "/classify will return 503.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_engine()
    yield


app = FastAPI(
    title="AutoVision API",
    description="Car make/model/generation recognition",
    version="0.2.0",
    lifespan=lifespan,
)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Ensure framework HTTP errors (404, 405, 413, ...) use the structured shape."""
    if exc.status_code >= 500:
        return _error(exc.status_code, "inference_failed", "Internal server error")
    return _error(exc.status_code, "bad_request", str(exc.detail))


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return _error(400, "bad_request", "Invalid request parameters")


@app.get("/health")
def health():
    return {
        "status": "ok" if engine is not None else "no_model",
        "model_loaded": engine is not None,
    }


@app.get("/version")
def version():
    return {
        "engine_version": ENGINE_VERSION,
        "model_version": engine.model_version if engine is not None else None,
        "taxonomy_version": engine.taxonomy_version if engine is not None else None,
    }


@app.get("/metadata")
def metadata():
    if engine is None:
        return {
            "model_loaded": False,
            "model_version": None,
            "taxonomy_version": None,
            "backend": None,
            "input_size": None,
            "num_classes": None,
            "num_makes": None,
            "quantization": None,
        }
    return {
        "model_loaded": True,
        "model_version": engine.model_version,
        "taxonomy_version": engine.taxonomy_version,
        "backend": engine.backend,
        "input_size": engine.input_size,
        "num_classes": engine.num_classes,
        "num_makes": engine.num_makes,
        "quantization": engine.quantization,
    }


def _serialize_prediction(p: Prediction) -> dict:
    return {
        "rank": p.rank,
        "class_name": p.class_name,
        "make": p.make,
        "model": p.model,
        "generation": p.generation,
        "year_start": p.year_start,
        "year_end": p.year_end,
        "rarity": p.rarity,
        "confidence": round(p.confidence, 4),
    }


def _serialize_result(result: ClassificationResult) -> dict:
    return {
        "predictions": [_serialize_prediction(p) for p in result.predictions],
        "top1": _serialize_prediction(result.top1) if result.top1 else None,
        "rejected": result.rejected,
        "rejection_reason": result.rejection_reason,
        "inference_ms": round(result.inference_ms, 1),
        "engine_version": result.engine_version,
        "model_version": result.model_version,
        "taxonomy_version": result.taxonomy_version,
    }


@app.post("/classify")
async def classify(
    image: UploadFile = File(...),
    top_k: int = Form(default=5),
):
    if engine is None:
        return _error(503, "model_not_loaded", "Model is not loaded")

    content_type = (image.content_type or "").lower()
    if content_type not in ACCEPTED_CONTENT_TYPES:
        return _error(
            400,
            "bad_request",
            "Unsupported content type. Accepted: image/jpeg, image/png, image/webp",
        )

    content = await image.read()
    if len(content) > MAX_UPLOAD_BYTES:
        return _error(
            413, "bad_request", "Image exceeds maximum upload size of 10 MB"
        )
    if not content:
        return _error(400, "bad_request", "Empty image upload")

    top_k = max(1, min(top_k, MAX_TOP_K))

    suffix = Path(image.filename or "image.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = engine.classify(tmp_path, top_k=top_k)
        return JSONResponse(content=_serialize_result(result))
    except InvalidImageError:
        return _error(422, "invalid_image", "Image could not be decoded")
    except Exception:
        return _error(500, "inference_failed", "Inference failed")
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
