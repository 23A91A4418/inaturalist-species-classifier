import importlib.util
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pytest

# Dynamically import script 04_evaluate_model
script_path = Path(__file__).parent.parent / "scripts" / "04_evaluate_model.py"
spec = importlib.util.spec_from_file_location("evaluate_model", script_path)
evaluate_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate_model)


def test_metric_calculation_correctness():
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 1, 0, 1, 2])

    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)

    assert round(acc, 4) == round(8 / 9, 4)
    assert precision > 0
    assert recall > 0
    assert f1 > 0

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    assert cm.shape == (3, 3)
    assert cm[0, 0] == 3  # All class 0 correctly predicted
