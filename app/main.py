"""FastAPI application for the lung disease detection project."""

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .inference import model_status, predict as run_prediction


BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

app = FastAPI(
    title="Lung Disease Detection API",
    description="Educational chest X-ray analysis demo. Not for clinical diagnosis.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    """Serve the browser interface."""

    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    """Report service readiness."""

    return {"status": "ok", "mode": model_status()}


@app.get("/model-info")
def model_info() -> dict[str, object]:
    """Describe the labels and model state exposed by this service."""

    return {
        "mode": model_status(),
        "labels": ["Normal", "Pneumonia"],
        "architecture": "ResNet-18" if model_status() == "trained" else None,
        "dataset": "Chest X-Ray Images (Pneumonia)",
        "research_only": True,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, object]:
    """Analyze an uploaded chest X-ray image using the demo pipeline."""

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Upload a PNG, JPEG, or WebP image.",
        )

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller.")

    try:
        result = run_prediction(content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "filename": file.filename or "uploaded-image",
        "prediction": result.label,
        "confidence": result.confidence,
        "mode": result.mode,
        "disclaimer": (
            "Demo output for research and education only. "
            "It is not a medical diagnosis."
        ),
    }
