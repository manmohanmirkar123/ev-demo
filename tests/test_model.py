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
    # Healthy-looking insulation (z < 2): a short model RUL alone must not raise an alarm.
    healthy = [(d, 3000.0 + (50 if d % 2 else -50)) for d in range(15)]
    assert score("insulation", healthy, model_rul=10.0)["status"] == "HEALTHY"
    # Every pack ages, so state of health skips the anomaly gate and the model RUL applies directly.
    soh = [(d, 84.0 - 0.2 * d) for d in range(15)]
    assert score("state_of_health", soh, model_rul=10.0)["status"] == "CRITICAL"
    # Model "nothing within the horizon" leaves the part healthy.
    assert score("state_of_health", [(d, 95.0) for d in range(15)], model_rul=119.0)["status"] == "HEALTHY"
