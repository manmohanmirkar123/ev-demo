"""Copy a trained model into the fleet account, where the scorer Lambda picks it up (≤5 min).

Usage: publish_model.py [path]   path = ev-pdm-model.json bundle (default: ml/.model/ev-pdm-model.json)
Same-account training (make train, make train-local) calls publish() directly; this CLI is for a model
trained in a separate ML account.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT = ROOT / "ml" / ".model" / "ev-pdm-model.json"


def load_config():
    return {k: v["value"] for k, v in json.loads((ROOT / "config.json").read_text()).items()}


def publish(s3, bucket, bundle):
    """bundle = {"info": {...}, "model": <XGBoost JSON model>}"""
    for key, body in (("model/model.json", bundle["model"]), ("model/model-info.json", bundle["info"])):
        s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(body).encode(), ContentType="application/json")
    print(f"published to s3://{bucket}/model/model.json  ({bundle['info'].get('trained_with', 'sagemaker')}, "
          f"{bundle['info']['training_job']})")
    print("The scorer switches to this model within 5 minutes.")


def main():
    import boto3

    path = pathlib.Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 and sys.argv[1] else DEFAULT
    if not path.exists():
        sys.exit(f"{path} not found — train first (make train / train-local / train-remote, or download "
                 "ev-pdm-model.json from CloudShell/Studio and pass FROM=<path>).")
    bundle = json.loads(path.read_text())
    if "model" not in bundle or "info" not in bundle:
        sys.exit(f"{path} is not an ev-pdm-model.json bundle")
    cfg = load_config()
    publish(boto3.Session(region_name=cfg["region"]).client("s3"), cfg["lake_bucket"], bundle)


if __name__ == "__main__":
    main()
