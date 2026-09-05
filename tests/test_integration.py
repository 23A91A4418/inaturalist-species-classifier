import os
import importlib.util
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).parent.parent


def test_required_artifacts_exist():
    required_files = [
        "Dockerfile",
        "docker-compose.yml",
        ".env.example",
        "requirements.txt",
        "README.md",
        ".gitignore"
    ]
    for file_name in required_files:
        path = REPO_ROOT / file_name
        assert path.exists(), f"Mandatory artifact missing: {file_name}"


def test_scripts_syntax_and_importability():
    scripts = [
        "01_fetch_data.py",
        "02_preprocess_data.py",
        "03_train_model.py",
        "04_evaluate_model.py",
        "05_run_grad_cam.py"
    ]
    for script in scripts:
        script_path = REPO_ROOT / "scripts" / script
        assert script_path.exists(), f"Script missing: {script}"

        spec = importlib.util.spec_from_file_location(script.replace(".py", ""), script_path)
        assert spec is not None, f"Failed to create module spec for {script}"
        mod = importlib.util.module_from_spec(spec)
        assert mod is not None, f"Failed to create module from spec for {script}"
