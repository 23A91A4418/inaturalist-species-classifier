#!/usr/bin/env python3
"""
scripts/04_evaluate_model.py
Model Evaluation Script for iNaturalist Species Classification Pipeline.
Evaluates the fine-tuned MobileNetV3 model on the test set.
Generates results/evaluation_metrics.json, results/confusion_matrix.png, and results/confusion_matrix.csv.
"""

import sys
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

PROCESSED_DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")


def get_mobilenet_v3_small(num_classes: int):
    """Load MobileNetV3 Small with matching classifier output size."""
    try:
        model = models.mobilenet_v3_small(weights=None)
    except TypeError:
        model = models.mobilenet_v3_small(pretrained=False)

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    class_map_path = PROCESSED_DATA_DIR / "class_map.json"
    if not class_map_path.exists():
        logging.error(f"class_map.json not found at {class_map_path}.")
        sys.exit(1)

    with open(class_map_path, "r") as f:
        class_map = json.load(f)

    # Inverse mapping: integer index to species name
    idx_to_class = {v: k for k, v in class_map.items()}
    class_names = [idx_to_class[i] for i in range(len(class_map))]
    num_classes = len(class_names)

    model_path = MODELS_DIR / "species_classifier.pth"
    if not model_path.exists():
        logging.error(f"Model file not found at {model_path}. Run 03_train_model.py first.")
        sys.exit(1)

    test_dir = PROCESSED_DATA_DIR / "test"
    if not test_dir.exists():
        logging.error(f"Test directory not found at {test_dir}.")
        sys.exit(1)

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(root=str(test_dir), transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading model state dict from {model_path} onto {device}...")

    model = get_mobilenet_v3_small(num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_targets = []

    logging.info("Running evaluation on test set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate metrics
    overall_acc = float(accuracy_score(all_targets, all_preds))
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)

    metrics = {
        "overall_accuracy": round(overall_acc, 4),
        "macro_precision": round(float(precision), 4),
        "macro_recall": round(float(recall), 4),
        "macro_f1_score": round(float(f1), 4)
    }

    logging.info("Test Evaluation Metrics:")
    logging.info(json.dumps(metrics, indent=2))

    metrics_path = RESULTS_DIR / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"Saved metrics to {metrics_path}")

    # Generate Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))
    
    # Save Confusion Matrix CSV with species names as row index and column headers
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_csv_path = RESULTS_DIR / "confusion_matrix.csv"
    cm_df.to_csv(cm_csv_path)
    logging.info(f"Saved confusion matrix CSV to {cm_csv_path}")

    # Plot and Save Confusion Matrix PNG
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix - iNaturalist Species Classification", fontsize=14)
    plt.xlabel("Predicted Species", fontsize=12)
    plt.ylabel("True Species", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_png_path = RESULTS_DIR / "confusion_matrix.png"
    plt.savefig(cm_png_path, dpi=300)
    plt.close()
    logging.info(f"Saved confusion matrix plot to {cm_png_path}")

    logging.info("Model evaluation complete!")


if __name__ == "__main__":
    main()
