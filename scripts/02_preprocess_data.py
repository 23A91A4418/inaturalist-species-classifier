#!/usr/bin/env python3
"""
scripts/02_preprocess_data.py
Data Preprocessing Script for iNaturalist Species Classification Pipeline.
Splits images into train (70%), val (15%), test (15%) and generates class_map.json.
"""

import os
import sys
import json
import shutil
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")


def main():
    if not RAW_DATA_DIR.exists():
        logging.error(f"Raw data directory {RAW_DATA_DIR} does not exist.")
        sys.exit(1)

    species_dirs = [d for d in RAW_DATA_DIR.iterdir() if d.is_dir()]
    species_names = sorted([d.name for d in species_dirs])
    
    if len(species_names) < 10:
        logging.warning(f"Found only {len(species_names)} species in {RAW_DATA_DIR}. Expected at least 10.")
        if not species_names:
            logging.error("No species directories found!")
            sys.exit(1)

    logging.info(f"Found {len(species_names)} species subdirectories.")

    # Generate class_map.json
    class_map = {species: idx for idx, species in enumerate(species_names)}
    class_map_file = PROCESSED_DATA_DIR / "class_map.json"
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(class_map_file, "w") as f:
        json.dump(class_map, f, indent=2)
    logging.info(f"Saved class mapping to {class_map_file}:")
    logging.info(json.dumps(class_map, indent=2))

    # Prepare train, val, test directories
    splits = ["train", "val", "test"]
    for split in splits:
        for species in species_names:
            (PROCESSED_DATA_DIR / split / species).mkdir(parents=True, exist_ok=True)

    all_raw_image_paths = []
    all_image_labels = []

    for species in species_names:
        species_dir = RAW_DATA_DIR / species
        # Accept common image extensions
        image_files = [f for f in species_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        for img in image_files:
            all_raw_image_paths.append(img)
            all_image_labels.append(species)

    total_images = len(all_raw_image_paths)
    logging.info(f"Total raw images collected across species: {total_images}")

    if total_images == 0:
        logging.error("No image files found in raw dataset.")
        sys.exit(1)

    # First split: Train (70%) vs Temp (30%)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_raw_image_paths,
        all_image_labels,
        test_size=0.30,
        random_state=42,
        stratify=all_image_labels
    )

    # Second split: Val (15% total -> 50% of temp) vs Test (15% total -> 50% of temp)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=0.50,
        random_state=42,
        stratify=temp_labels
    )

    split_data = {
        "train": (train_paths, train_labels),
        "val": (val_paths, val_labels),
        "test": (test_paths, test_labels)
    }

    total_processed = 0
    for split_name, (paths, labels) in split_data.items():
        logging.info(f"Processing split '{split_name}': {len(paths)} images ({len(paths)/total_images*100:.1f}%)")
        for src_path, label in zip(paths, labels):
            dest_dir = PROCESSED_DATA_DIR / split_name / label
            dest_path = dest_dir / src_path.name
            shutil.copy2(src_path, dest_path)
            total_processed += 1

    logging.info(f"Successfully processed {total_processed}/{total_images} images into {PROCESSED_DATA_DIR}.")


if __name__ == "__main__":
    main()
