"""XGBoost days-to-limit model.

Runs as the SageMaker training-job entry point (XGBoost framework container, script mode) and locally
from ml/evaluate.py, so both train the identical model. Saves model.json, which the scorer Lambda
evaluates in pure Python (lambda/scorer/tree_model.py).
"""
import argparse
import glob
import os

import numpy as np
import xgboost as xgb

from hyperparams import HYPERPARAMS


def load(path):
    """A CSV file, or a directory of them (a SageMaker channel). Label is column 0."""
    files = sorted(glob.glob(os.path.join(path, "*.csv"))) if os.path.isdir(path) else [path]
    return np.vstack([np.loadtxt(f, delimiter=",", ndmin=2) for f in files])


def train(train_path, val_path, hp=None, verbose=False):
    hp = {**HYPERPARAMS, **(hp or {})}
    tr, va = load(train_path), load(val_path)
    params = {k: v for k, v in hp.items() if k not in ("num_round", "early_stopping_rounds")}
    dtrain = xgb.DMatrix(tr[:, 1:], label=tr[:, 0])
    dval = xgb.DMatrix(va[:, 1:], label=va[:, 0])
    booster = xgb.train(params, dtrain, int(hp["num_round"]), evals=[(dval, "validation")],
                        early_stopping_rounds=int(hp["early_stopping_rounds"]),
                        verbose_eval=10 if verbose else False)
    booster = booster[: booster.best_iteration + 1]  # keep only the trees up to the best round
    return booster, va, booster.predict(dval)


def report(va, pred):
    err = np.abs(pred - va[:, 0])
    near = va[:, 0] <= 60
    return {"validation_mae": float(err.mean()), "near_failure_mae": float(err[near].mean())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for name, default in HYPERPARAMS.items():
        ap.add_argument(f"--{name}", type=type(default), default=default)
    ap.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN"))
    ap.add_argument("--validation", default=os.environ.get("SM_CHANNEL_VALIDATION"))
    ap.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "."))
    args = ap.parse_args()

    booster, va, pred = train(args.train, args.validation, {k: getattr(args, k) for k in HYPERPARAMS}, verbose=True)
    metrics = report(va, pred)
    # Matched by the training job's MetricDefinitions, so they show up in the SageMaker console.
    print(f"final validation-mae:{metrics['validation_mae']:.4f}")
    print(f"final near-failure-mae:{metrics['near_failure_mae']:.4f}")
    booster.save_model(os.path.join(args.model_dir, "model.json"))
