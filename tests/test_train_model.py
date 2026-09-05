import importlib.util
from pathlib import Path
import torch
import torch.nn as nn
import pytest

# Dynamically import script 03_train_model
script_path = Path(__file__).parent.parent / "scripts" / "03_train_model.py"
spec = importlib.util.spec_from_file_location("train_model", script_path)
train_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_model)


def test_get_mobilenet_v3_small_architecture():
    num_classes = 10
    model = train_model.get_mobilenet_v3_small(num_classes)

    assert isinstance(model, nn.Module)
    # Check that final linear layer has output dimension equal to num_classes
    assert model.classifier[3].out_features == num_classes

    # Test forward pass with dummy tensor
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)

    assert output.shape == (2, num_classes)
