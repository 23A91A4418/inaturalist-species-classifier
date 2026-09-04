#!/usr/bin/env python3
"""
scripts/03_train_model.py
Model Training Script for iNaturalist Species Classification Pipeline.
Fine-tunes a pre-trained MobileNetV3-Small model on the dataset using PyTorch.
Tracks metrics to results/training_log.csv and saves weights to models/species_classifier.pth.
"""

import os
import sys
import json
import csv
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

PROCESSED_DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")

EPOCHS = int(os.getenv("EPOCHS", 5))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.001))


def get_mobilenet_v3_small(num_classes: int):
    """Load pre-trained MobileNetV3 Small and replace the classifier head."""
    try:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
    except AttributeError:
        # Fallback for older torchvision versions
        model = models.mobilenet_v3_small(pretrained=True)

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    class_map_path = PROCESSED_DATA_DIR / "class_map.json"
    if not class_map_path.exists():
        logging.error(f"class_map.json not found at {class_map_path}. Run 02_preprocess_data.py first.")
        sys.exit(1)

    with open(class_map_path, "r") as f:
        class_map = json.load(f)

    num_classes = len(class_map)
    logging.info(f"Loaded class map with {num_classes} species.")

    # Define Data Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets and DataLoaders
    train_dir = PROCESSED_DATA_DIR / "train"
    val_dir = PROCESSED_DATA_DIR / "val"

    if not train_dir.exists() or not val_dir.exists():
        logging.error("Processed train or val directory missing. Ensure 02_preprocess_data.py was executed.")
        sys.exit(1)

    train_dataset = datasets.ImageFolder(root=str(train_dir), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=str(val_dir), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    logging.info(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model = get_mobilenet_v3_small(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    training_log_path = RESULTS_DIR / "training_log.csv"
    with open(training_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy"])

    best_val_acc = 0.0
    model_save_path = MODELS_DIR / "species_classifier.pth"

    logging.info(f"Starting model training for {EPOCHS} epochs...")

    for epoch in range(1, EPOCHS + 1):
        # Training Phase
        model.train()
        running_train_loss = 0.0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)
            train_total += images.size(0)

        epoch_train_loss = running_train_loss / train_total

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += images.size(0)

        epoch_val_loss = running_val_loss / val_total
        epoch_val_acc = val_correct / val_total

        logging.info(
            f"Epoch [{epoch}/{EPOCHS}] - Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}"
        )

        # Log to CSV
        with open(training_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, round(epoch_train_loss, 4), round(epoch_val_loss, 4), round(epoch_val_acc, 4)])

        # Save best model
        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"Saved new best model checkpoint to {model_save_path} (Val Acc: {best_val_acc:.4f})")

    logging.info(f"Training completed! Best Validation Accuracy: {best_val_acc:.4f}")
    logging.info(f"Training log written to {training_log_path}")


if __name__ == "__main__":
    main()
