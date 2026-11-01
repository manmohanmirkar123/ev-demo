"""The SageMaker entry point trains a model the Lambda's pure-Python evaluator reproduces exactly."""
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "ml"), str(ROOT / "lambda" / "scorer"), str(ROOT / "simulator")]

from health import FEATURES, score  # noqa: E402
from tree_model import TreeModel  # noqa: E402


def test_entry_point_model_matches_xgboost(tmp_path):
    xgb = pytest.importorskip("xgboost")
    np = pytest.importorskip("numpy")
    from dataset import build, write_csv

    write_csv(build(60, seed=1), tmp_path / "train" / "train.csv")
    write_csv(build(20, seed=2), tmp_path / "validation" / "validation.csv")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    # Invoke it the way the SageMaker training toolkit does: channels via env, hyperparameters as CLI args.
    env = {**os.environ, "SM_CHANNEL_TRAIN": str(tmp_path / "train"),
           "SM_CHANNEL_VALIDATION": str(tmp_path / "validation"), "SM_MODEL_DIR": str(model_dir)}
    out = subprocess.run([sys.executable, str(ROOT / "ml" / "sm_code" / "xgb_train.py"),
                          "--max_depth", "4", "--num_round", "50"],
                         env=env, capture_output=True, text=True, check=True).stdout
    assert "final validation-mae:" in out

    path = model_dir / "model.json"
    tree = TreeModel(path.read_text())
    assert tree.num_features == len(FEATURES)

    booster = xgb.Booster()
    booster.load_model(str(path))
    val = np.loadtxt(tmp_path / "validation" / "validation.csv", delimiter=",")
    expected = booster.predict(xgb.DMatrix(val[:, 1:]))
    got = np.array([tree.predict(list(r)) for r in val[:, 1:]])
    assert np.abs(got - expected).max() < 1e-3


def test_model_rul_needs_visible_deviation():
    # Healthy-looking tyre (z < 2): a short model RUL alone must not raise an alarm.
    healthy = [(d, 36.0 + (0.1 if d % 2 else -0.1)) for d in range(15)]
    assert score("tyres", healthy, model_rul=10.0)["status"] == "HEALTHY"
    # Brake pads wear by design, so the model RUL applies without the anomaly gate.
    pads = [(d, 4.0 - 0.05 * d) for d in range(15)]
    assert score("brake_pads", pads, model_rul=10.0)["status"] == "CRITICAL"
    # Model "nothing within the horizon" leaves the part healthy.
    assert score("brake_pads", [(d, 9.0) for d in range(15)], model_rul=119.0)["status"] == "HEALTHY"
