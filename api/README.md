# AutoVision API

Self-hosted REST API for car recognition.

## Quick Start

```bash
# Build
docker build -t autovision-api -f api/Dockerfile .

# Run (mount your model directory)
docker run -p 8000:8000 -v /path/to/models:/app/models autovision-api
```

## Endpoints

### `GET /health`

Health check. Returns model load status.

### `POST /classify`

Classify a car image.

**Request:**
- `image` (file): Image upload (JPEG, PNG)
- `top_k` (form, optional): Number of predictions (default: 5, max: 50)

**Response:**
```json
{
  "predictions": [
    {
      "rank": 1,
      "class_name": "bmw_3_series_f30",
      "make": "BMW",
      "model": "3 Series",
      "generation": "F30",
      "year_start": 2012,
      "year_end": 2018,
      "rarity": "COMMON",
      "confidence": 0.9234
    }
  ],
  "inference_ms": 45.2
}
```

## Configuration

Environment variables:
- `AUTOVISION_MODEL` — path to `.tflite` model (default: `/app/models/car_classifier.tflite`)
- `AUTOVISION_CLASSES` — path to `class_mapping.json` (default: `/app/models/class_mapping.json`)
