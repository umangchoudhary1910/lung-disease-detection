# Lung Disease Detection

This project turns the supplied project proposal into a Python API with a
browser interface. It accepts PNG, JPEG, and WebP chest X-ray uploads and
classifies them as **Normal** or **Pneumonia**.

## Important limitation

The public dataset selected for this project supports only Normal and
Pneumonia. Tuberculosis and Lung Cancer are not displayed because this dataset
does not contain valid labels for them. The API uses a trained ResNet-18 model
when `models/pneumonia_resnet18.pt` exists, and visibly falls back to a
deterministic demo pipeline otherwise.

## Run locally

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>. Interactive API documentation is available at
<http://127.0.0.1:8000/docs>.

## Test

```powershell
pytest
```

## Download and train

The training scripts use the public **Chest X-Ray Images (Pneumonia)** dataset
by Paul Mooney. Configure a Kaggle API token first, then run:

```powershell
pip install kagglehub
python scripts/download_dataset.py
python scripts/train_model.py
python scripts/evaluate_model.py
```

Arrange the downloaded data as `data/chest_xray/train`, `data/chest_xray/val`,
and `data/chest_xray/test`, each containing `NORMAL` and `PNEUMONIA` folders.
Do not commit the dataset or trained weights.

The API exposes `GET /model-info` so the UI and integrations can verify
whether trained weights are active. Never interpret the confidence score as a
clinical probability.

## API

- `GET /health` checks readiness.
- `POST /predict` accepts a multipart `file` upload and returns the predicted
  category, confidence, mode, filename, and disclaimer.

Supported categories are Normal and Pneumonia.
