"""ML without SageMaker: train the same XGBoost model on this machine and publish it to the fleet account.

Same data, same entry point (ml/sm_code/xgb_train.py) and same model format as the SageMaker job, so the
scorer Lambda can't tell them apart — only the dashboard banner says where it was trained.
"""
import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "ml"), str(ROOT / "ml" / "sm_code"), str(ROOT / "lambda" / "scorer")]

from dataset import build_splits  # noqa: E402
from health import FEATURES  # noqa: E402
from publish_model import load_config, publish  # noqa: E402
from xgb_train import report, train  # noqa: E402

MODEL_DIR = ROOT / "ml" / ".model"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-publish", action="store_true", help="train and save only")
    args = ap.parse_args()

    print("building run-to-failure training data from simulated fleet history…")
    train_csv, val_csv, n_tr, n_va = build_splits(ROOT / "ml" / ".data")
    started = time.time()
    booster, val, pred = train(train_csv, val_csv)
    metrics = report(val, pred)
    print(f"  {n_tr} train rows, {n_va} validation rows, {booster.num_boosted_rounds()} trees, "
          f"{time.time() - started:.1f}s")
    print(f"  validation MAE {metrics['validation_mae']:.2f} days "
          f"(parts failing within 60 days: {metrics['near_failure_mae']:.2f})")

    info = {
        "trained_with": "local", "training_job": f"local-{time.strftime('%Y%m%d-%H%M%S')}",
        "metrics": {"validation:mae": metrics["validation_mae"],
                    "validation:near_failure_mae": metrics["near_failure_mae"]},
        "features": FEATURES, "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    bundle = {"info": info, "model": json.loads(booster.save_raw("json"))}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "ev-pdm-model.json").write_text(json.dumps(bundle))
    print(f"saved {MODEL_DIR / 'ev-pdm-model.json'}")

    if not args.no_publish:
        import boto3

        cfg = load_config()
        publish(boto3.Session(region_name=cfg["region"]).client("s3"), cfg["lake_bucket"], bundle)


if __name__ == "__main__":
    main()
