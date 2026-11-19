from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


client = TestClient(app)


def image_bytes() -> bytes:
    image = Image.new("RGB", (32, 32), color="white")
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "mode": "demo"}


def test_model_info_describes_supported_labels() -> None:
    response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json()["labels"] == ["Normal", "Pneumonia"]
    assert response.json()["research_only"] is True


def test_predict_returns_demo_contract() -> None:
    response = client.post(
        "/predict",
        files={"file": ("xray.png", image_bytes(), "image/png")},
    )
    body = response.json()
    assert response.status_code == 200
    assert body["mode"] == "demo"
    assert body["prediction"] in {"Normal", "Pneumonia"}
    assert 0.55 <= body["confidence"] <= 0.9
    assert "not a medical diagnosis" in body["disclaimer"]


def test_predict_rejects_non_image_content_type() -> None:
    response = client.post(
        "/predict",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415
