"""IoT Rule -> Lambda: score every component of one telemetry message, store health, raise risk events."""
import json
import os
import time
from decimal import Decimal

import boto3

from health import COMPONENTS, RUL_CAP, SEVERITY, extract, features, score
from tree_model import TreeModel

HEALTH_TABLE = os.environ["HEALTH_TABLE"]
BASELINE_BUCKET = os.environ["BASELINE_BUCKET"]
BASELINE_KEY = os.environ.get("BASELINE_KEY", "baseline/baseline.json")
MODEL_KEY = os.environ.get("MODEL_KEY", "model/model.json")
EVENT_SOURCE = "ev.pdm"
WINDOW = 30
ARTIFACT_TTL = 300

ddb = boto3.resource("dynamodb")
table = ddb.Table(HEALTH_TABLE)
events = boto3.client("events")
s3 = boto3.client("s3")

_cache = {}


def artifact(key, parse):
    """S3 JSON artifact, reloaded every 5 min; None until it has been published."""
    entry = _cache.get(key)
    if entry is None or time.time() - entry[0] > ARTIFACT_TTL:
        try:
            body = s3.get_object(Bucket=BASELINE_BUCKET, Key=key)["Body"].read()
            entry = (time.time(), parse(body))
        except s3.exceptions.NoSuchKey:
            entry = (time.time(), None)
        _cache[key] = entry
    return entry[1]


def baseline():
    """Fleet baseline written by ml/build_baseline.py; falls back to the defaults in health.py."""
    return artifact(BASELINE_KEY, json.loads) or {}


def model():
    """XGBoost days-to-limit model trained on SageMaker (ml/train_sagemaker.py); None -> linear trend."""
    return artifact(MODEL_KEY, TreeModel)


def to_ddb(obj):
    return json.loads(json.dumps(obj), parse_float=Decimal)


def handler(event, context):
    vehicle_id = event["vehicle_id"]
    day = int(event["sim_day"])
    values = extract(event)
    learned = baseline()
    rul_model = model()

    keys = [{"vehicle_id": vehicle_id, "component": c} for c in values]
    resp = ddb.batch_get_item(RequestItems={HEALTH_TABLE: {"Keys": keys}})
    existing = {i["component"]: i for i in resp["Responses"].get(HEALTH_TABLE, [])}

    risk_events = []
    with table.batch_writer() as writer:
        for component, value in values.items():
            prev = existing.get(component, {})
            run_id = event.get("run_id")
            if run_id is not None and prev.get("run_id") is not None:
                restarted = prev["run_id"] != run_id  # a new simulator run: start fresh
            else:
                restarted = prev.get("sim_day", -1) >= day  # no run ids: day going backwards means a replay
            # IoT -> Lambda is asynchronous, so days can arrive out of order: merge by day, don't append.
            merged = {} if restarted else {int(d): float(v) for d, v in prev.get("history", [])}
            merged[day] = value
            history = sorted(merged.items())[-WINDOW:]
            latest_day = history[-1][0]
            predicted = None
            if rul_model is not None:
                x = features(component, history, learned.get(component))
                if x is not None:
                    predicted = min(RUL_CAP, max(0.0, rul_model.predict(x)))
            result = score(component, history, learned.get(component), model_rul=predicted)
            spec = COMPONENTS[component]
            # Peak status gives hysteresis: a part hovering at a boundary raises one work order, not many.
            peak = "HEALTHY" if restarted else prev.get("peak_status", "HEALTHY")
            escalated = SEVERITY[result["status"]] > SEVERITY[peak]
            if escalated:
                peak = result["status"]

            writer.put_item(Item=to_ddb({
                "vehicle_id": vehicle_id,
                "component": component,
                "label": spec["label"],
                "unit": spec["unit"],
                "threshold": spec["threshold"],
                "sim_day": latest_day,
                "odometer_km": (event.get("odometer_km") if day == latest_day or "odometer_km" not in prev
                                else float(prev["odometer_km"])),
                "history": history,
                "updated_at": event.get("ts"),
                "peak_status": peak,
                "run_id": run_id,
                **result,
            }))

            if escalated:
                risk_events.append({
                    "Source": EVENT_SOURCE,
                    "DetailType": "ComponentAtRisk",
                    "Detail": json.dumps({
                        "vehicle_id": vehicle_id,
                        "component": component,
                        "label": spec["label"],
                        "unit": spec["unit"],
                        "sim_day": day,
                        "odometer_km": event.get("odometer_km"),
                        "rul_text": "unknown" if result["rul_days"] is None else f"~{result['rul_days']:.0f}",
                        **result,
                    }),
                })

    if risk_events:
        events.put_events(Entries=risk_events)
    return {"vehicle_id": vehicle_id, "sim_day": day, "risk_events": len(risk_events)}
