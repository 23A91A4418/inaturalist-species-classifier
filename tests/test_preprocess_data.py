import json
import importlib.util
from pathlib import Path
from unittest.mock import patch
import pytest

# Dynamically import script 02_preprocess_data
script_path = Path(__file__).parent.parent / "scripts" / "02_preprocess_data.py"
spec = importlib.util.spec_from_file_location("preprocess_data", script_path)
preprocess_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preprocess_data)


def test_preprocess_data_flow(tmp_path):
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"

    # Create fake species directories and images
    species_names = ["Species_A", "Species_B"]
    for species in species_names:
        s_dir = raw_dir / species
        s_dir.mkdir(parents=True, exist_ok=True)
        for i in range(10):
            (s_dir / f"image_{i}.jpg").write_bytes(b"dummy_image_data")

    with patch.object(preprocess_data, "RAW_DATA_DIR", raw_dir), \
         patch.object(preprocess_data, "PROCESSED_DATA_DIR", processed_dir):
        preprocess_data.main()

    # Verify class_map.json
    class_map_file = processed_dir / "class_map.json"
    assert class_map_file.exists()

    with open(class_map_file, "r") as f:
        class_map = json.load(f)

    assert "Species_A" in class_map
    assert "Species_B" in class_map
    assert len(class_map) == 2

    # Verify directory structure for train, val, test
    for split in ["train", "val", "test"]:
        for species in species_names:
            split_species_dir = processed_dir / split / species
            assert split_species_dir.exists()
            assert len(list(split_species_dir.glob("*.jpg"))) > 0
