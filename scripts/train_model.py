"""Train a binary chest X-ray classifier on an ImageFolder dataset."""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


DATA_DIR = Path("data/chest_xray")
MODEL_DIR = Path("models")
EPOCHS = 3
BATCH_SIZE = 32

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_data = datasets.ImageFolder(DATA_DIR / "train", transform=transform)
validation_data = datasets.ImageFolder(DATA_DIR / "val", transform=transform)
if train_data.classes != ["NORMAL", "PNEUMONIA"]:
    raise ValueError(f"Expected NORMAL and PNEUMONIA folders, found {train_data.classes}")

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size=BATCH_SIZE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_function = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss_function(model(images), labels).backward()
        optimizer.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in validation_loader:
            predictions = model(images.to(device)).argmax(dim=1)
            correct += int((predictions == labels.to(device)).sum())
            total += labels.size(0)
    print(f"epoch={epoch + 1} validation_accuracy={correct / max(total, 1):.3f}")

MODEL_DIR.mkdir(exist_ok=True)
torch.save(model.cpu().state_dict(), MODEL_DIR / "pneumonia_resnet18.pt")
print("Saved models/pneumonia_resnet18.pt")
