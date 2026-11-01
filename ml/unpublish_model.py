"""Remove the model from the fleet account so the scorer falls back to the linear trend (for A/B demos)."""
import json
import pathlib

import boto3

ROOT = pathlib.Path(__file__).resolve().parent.parent
cfg = {k: v["value"] for k, v in json.loads((ROOT / "config.json").read_text()).items()}
s3 = boto3.Session(region_name=cfg["region"]).client("s3")
s3.delete_objects(Bucket=cfg["lake_bucket"],
                  Delete={"Objects": [{"Key": "model/model.json"}, {"Key": "model/model-info.json"}]})
print("model removed; the scorer falls back to the linear trend within 5 minutes.")
