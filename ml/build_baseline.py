"""Learn the healthy-fleet baseline from the data lake with Athena and publish it for the scorer.

Uses the fleet's first N simulated days (the "commissioning" window, before faults develop) to compute
mean and std per health indicator. The scorer Lambda reloads baseline/baseline.json every 5 minutes.
"""
import argparse
import json
import pathlib
import sys
import time

import boto3

ROOT = pathlib.Path(__file__).resolve().parent.parent

QUERY = """
SELECT
  avg((battery.cell_v_max - battery.cell_v_min) * 1000)      AS cell_imbalance_mean,
  stddev((battery.cell_v_max - battery.cell_v_min) * 1000)   AS cell_imbalance_std,
  avg(battery.cell_temp_max_c - battery.coolant_temp_c)      AS pack_cooling_mean,
  stddev(battery.cell_temp_max_c - battery.coolant_temp_c)   AS pack_cooling_std,
  avg(battery.insulation_kohm)                               AS insulation_mean,
  stddev(battery.insulation_kohm)                            AS insulation_std,
  count(*) AS samples
FROM telemetry
WHERE sim_day < {days}
"""


def run_query(athena, sql, database, workgroup):
    qid = athena.start_query_execution(
        QueryString=sql, QueryExecutionContext={"Database": database}, WorkGroup=workgroup
    )["QueryExecutionId"]
    while True:
        state = athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"]
        if state["State"] in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        time.sleep(1)
    if state["State"] != "SUCCEEDED":
        sys.exit(f"Athena query {state['State']}: {state.get('StateChangeReason')}")
    rows = athena.get_query_results(QueryExecutionId=qid)["ResultSet"]["Rows"]
    header = [c["VarCharValue"] for c in rows[0]["Data"]]
    return dict(zip(header, [c.get("VarCharValue") for c in rows[1]["Data"]]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=10, help="commissioning window in simulated days")
    args = ap.parse_args()

    cfg = {k: v["value"] for k, v in json.loads((ROOT / "config.json").read_text()).items()}
    if not cfg["athena_workgroup"]:
        sys.exit("Data lake is disabled (enable_data_lake=false); the scorer uses built-in defaults.")

    session = boto3.Session(region_name=cfg["region"])
    row = run_query(session.client("athena"), QUERY.format(days=args.days),
                    cfg["glue_database"], cfg["athena_workgroup"])
    if not row.get("samples") or int(row["samples"]) == 0:
        sys.exit("No telemetry in the lake yet — Firehose buffers ~60s after the simulator starts.")

    baseline = {
        comp: {"mean": round(float(row[f"{comp}_mean"]), 4), "std": round(float(row[f"{comp}_std"]), 4)}
        for comp in ("cell_imbalance", "pack_cooling", "insulation")
    }
    session.client("s3").put_object(
        Bucket=cfg["lake_bucket"], Key="baseline/baseline.json",
        Body=json.dumps(baseline, indent=2).encode(), ContentType="application/json",
    )
    print(f"Baseline from {row['samples']} samples (sim days 0–{args.days - 1}):")
    print(json.dumps(baseline, indent=2))
    print(f"Published to s3://{cfg['lake_bucket']}/baseline/baseline.json")


if __name__ == "__main__":
    main()
