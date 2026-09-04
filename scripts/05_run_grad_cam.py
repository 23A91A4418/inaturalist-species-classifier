#!/usr/bin/env python3
"""
scripts/05_run_grad_cam.py
Grad-CAM Model Interpretation Script for iNaturalist Species Classification Pipeline.
Generates visual heatmap explanations for model predictions on sample test images.
Saves output overlay images to results/grad_cam_outputs/.
"""

import sys
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

PROCESSED_DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
GRAD_CAM_DIR = RESULTS_DIR / "grad_cam_outputs"


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        score = output[0, target_class]
        score.backward()

        # Global Average Pooling of gradients across spatial dimensions
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        # Weighted sum of feature maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize CAM to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize CAM to match input image dimensions (224, 224)
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        heatmap = cam.squeeze().cpu().numpy()

        return heatmap, target_class, output


def get_mobilenet_v3_small(num_classes: int):
    """Load MobileNetV3 Small with matching output size."""
    try:
        model = models.mobilenet_v3_small(weights=None)
    except TypeError:
        model = models.mobilenet_v3_small(pretrained=False)

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def main():
    GRAD_CAM_DIR.mkdir(parents=True, exist_ok=True)

    class_map_path = PROCESSED_DATA_DIR / "class_map.json"
    if not class_map_path.exists():
        logging.error(f"class_map.json not found at {class_map_path}.")
        sys.exit(1)

    with open(class_map_path, "r") as f:
        class_map = json.load(f)

    idx_to_class = {v: k for k, v in class_map.items()}
    num_classes = len(class_map)

    model_path = MODELS_DIR / "species_classifier.pth"
    if not model_path.exists():
        logging.error(f"Model checkpoint not found at {model_path}.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading trained model onto {device}...")

    model = get_mobilenet_v3_small(num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Target layer for Grad-CAM in MobileNetV3 is the last convolutional block in features
    target_layer = model.features[-1]
    logging.info(f"Registered Grad-CAM hooks on target layer: {target_layer}")

    grad_cam = GradCAM(model, target_layer)

    test_dir = PROCESSED_DATA_DIR / "test"
    if not test_dir.exists():
        logging.error(f"Test dataset directory {test_dir} not found.")
        sys.exit(1)

    # Collect sample test images (at least 1 per species, up to 10 total)
    sample_images = []
    for species_dir in sorted(test_dir.iterdir()):
        if species_dir.is_dir():
            images = [f for f in species_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            if images:
                sample_images.append((images[0], species_dir.name))

    if len(sample_images) < 5:
        # Fallback if fewer species dirs exist
        all_imgs = list(test_dir.glob("*/*.jpg"))
        sample_images = [(img, img.parent.name) for img in all_imgs[:10]]

    logging.info(f"Generating Grad-CAM visualizations for {len(sample_images)} test images...")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    saved_count = 0
    for idx, (img_path, true_species) in enumerate(sample_images, 1):
        try:
            raw_img = Image.open(img_path).convert("RGB")
            resized_raw_img = raw_img.resize((224, 224))
            
            input_tensor = transform(raw_img).unsqueeze(0).to(device)
            heatmap, pred_class_idx, logits = grad_cam.generate_heatmap(input_tensor)
            pred_species = idx_to_class.get(pred_class_idx, f"Class {pred_class_idx}")

            # Plot Side-by-Side: Original Image vs Grad-CAM Overlay
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            
            axes[0].imshow(resized_raw_img)
            axes[0].set_title(f"True: {true_species}", fontsize=11)
            axes[0].axis("off")

            axes[1].imshow(resized_raw_img)
            # Overlay heatmap with jet colormap and transparency
            axes[1].imshow(heatmap, cmap="jet", alpha=0.5)
            axes[1].set_title(f"Pred: {pred_species}\n(Grad-CAM Heatmap)", fontsize=11, color="green" if pred_species == true_species else "red")
            axes[1].axis("off")

            plt.suptitle(f"Grad-CAM Explanation - Sample #{idx}", fontsize=14, fontweight="bold")
            plt.tight_layout()

            output_file = GRAD_CAM_DIR / f"grad_cam_sample_{idx}.png"
            plt.savefig(output_file, dpi=200, bbox_inches="tight")
            plt.close()

            saved_count += 1
            logging.info(f"Saved [{saved_count}] Grad-CAM output: {output_file.name} (True: {true_species} | Pred: {pred_species})")
        except Exception as e:
            logging.error(f"Error generating Grad-CAM for {img_path}: {e}")

    logging.info(f"Successfully generated {saved_count} Grad-CAM visualizations in {GRAD_CAM_DIR}")


if __name__ == "__main__":
    main()
