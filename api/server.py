"""AutoVision API server — car recognition via HTTP."""

import os
import sys
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# Add SDK to path
sys.path.insert(0, str(Path(__file__).parent.parent / "sdk" / "python"))

from autovision import AutoVision

app = FastAPI(
    title="AutoVision API",
    description="Car make/model/generation recognition",
    version="0.1.0",
)

# Load model at startup
MODEL_PATH = os.environ.get("AUTOVISION_MODEL", "/app/models/car_classifier.tflite")
CLASS_MAPPING_PATH = os.environ.get("AUTOVISION_CLASSES", "/app/models/class_mapping.json")

engine: AutoVision | None = None


@app.on_event("startup")
def load_model():
    global engine
    if not Path(MODEL_PATH).exists():
        print(f"WARNING: Model not found at {MODEL_PATH}. /classify will return 503.")
        return
    engine = AutoVision(
        model_path=MODEL_PATH,
        class_mapping_path=CLASS_MAPPING_PATH if Path(CLASS_MAPPING_PATH).exists() else None,
    )
    print(f"Model loaded from {MODEL_PATH}")


@app.get("/health")
def health():
    return {
        "status": "ok" if engine is not None else "no_model",
        "model_path": MODEL_PATH,
    }


@app.post("/classify")
async def classify(
    image: UploadFile = File(...),
    top_k: int = Form(default=5),
):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if top_k < 1 or top_k > 50:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 50")

    # Save upload to temp file
    suffix = Path(image.filename or "image.jpg").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await image.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        start = time.time()
        results = engine.classify(tmp_path, top_k=top_k)
        elapsed_ms = (time.time() - start) * 1000

        return JSONResponse(
            content={
                "predictions": [
                    {
                        "rank": i + 1,
                        "class_name": p.class_name,
                        "make": p.make,
                        "model": p.model,
                        "generation": p.generation,
                        "year_start": p.year_start,
                        "year_end": p.year_end,
                        "rarity": p.rarity,
                        "confidence": round(p.confidence, 4),
                    }
                    for i, p in enumerate(results)
                ],
                "inference_ms": round(elapsed_ms, 1),
            }
        )
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
