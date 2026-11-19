"""Inference services for the lung disease detection demo."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError


LABELS = ("Normal", "Pneumonia")
MODEL_PATH = Path(__file__).parent.parent / "models" / "pneumonia_resnet18.pt"


@dataclass(frozen=True)
class Prediction:
    """A model prediction returned by the API."""

    label: str
    confidence: float
    mode: str


def validate_image(content: bytes) -> tuple[int, int]:
    """Validate an uploaded image and return its dimensions."""

    if not content:
        raise ValueError("The uploaded file is empty.")

    try:
        with Image.open(BytesIO(content)) as image:
            image.verify()
        with Image.open(BytesIO(content)) as image:
            return image.size
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc


def predict_demo(content: bytes) -> Prediction:
    """Return a deterministic demo prediction for an uploaded image.

    This intentionally does not claim clinical accuracy. A trained model can
    replace this function without changing the API contract.
    """

    validate_image(content)
    digest = hashlib.sha256(content).digest()
    label = LABELS[digest[0] % len(LABELS)]
    confidence = round(0.55 + (digest[1] / 255) * 0.35, 3)
    return Prediction(label=label, confidence=confidence, mode="demo")


@lru_cache(maxsize=1)
def _load_model():
    """Load trained weights once per process."""

    import torch
    from torchvision import models

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(LABELS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    return model


def model_status() -> str:
    """Return whether trained weights are available."""

    return "trained" if MODEL_PATH.exists() else "demo"


def predict(content: bytes) -> Prediction:
    """Run the trained model when weights exist, otherwise use demo mode."""

    validate_image(content)
    if not MODEL_PATH.exists():
        return predict_demo(content)

    try:
        import torch
        from torchvision import transforms
    except ImportError as exc:
        raise RuntimeError("Training dependencies are not installed.") from exc

    model = _load_model()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    with Image.open(BytesIO(content)).convert("RGB") as image:
        with torch.no_grad():
            probabilities = torch.softmax(model(transform(image).unsqueeze(0)), dim=1)[0]
    index = int(torch.argmax(probabilities))
    return Prediction(
        label=LABELS[index],
        confidence=round(float(probabilities[index]), 3),
        mode="trained",
    )
