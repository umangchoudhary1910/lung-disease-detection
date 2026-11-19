"""Evaluate trained weights on data/chest_xray/test and print key metrics."""

from pathlib import Path

import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


data_dir = Path("data/chest_xray/test")
model_path = Path("models/pneumonia_resnet18.pt")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
dataset = datasets.ImageFolder(data_dir, transform=transform)
loader = DataLoader(dataset, batch_size=32)
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
model.eval()

actual = []
predicted = []
with torch.no_grad():
    for images, labels in loader:
        predicted.extend(model(images).argmax(dim=1).tolist())
        actual.extend(labels.tolist())

print(classification_report(actual, predicted, target_names=dataset.classes, zero_division=0))
