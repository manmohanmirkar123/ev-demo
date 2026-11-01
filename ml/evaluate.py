"""Train the same XGBoost model locally and compare it with the linear trend on the demo fleet.

Answers "is the model worth deploying?" before spending sandbox time on SageMaker:
lead time per injected fault (warning day vs. limit day) and false alarms on healthy parts.
"""
import pathlib
import sys

import numpy as np
import xgboost as xgb

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "ml"), str(ROOT / "ml" / "sm_code"), str(ROOT / "lambda" / "scorer"),
                str(ROOT / "simulator")]

from dataset import build_splits, crossed  # noqa: E402
from fleet import FAULTS, make_fleet  # noqa: E402
from health import COMPONENTS, extract, features, score  # noqa: E402
from tree_model import TreeModel  # noqa: E402
from xgb_train import report, train  # noqa: E402

DAYS = 120
WINDOW = 30
FAULT_COMPONENT = {"weak_cell": "cell_imbalance", "cooling_plate": "pack_cooling",
                   "moisture_ingress": "insulation", "fast_fade": "state_of_health"}


def train_local(data_dir):
    tr, va, n_tr, n_va = build_splits(data_dir)
    booster, val, pred = train(tr, va)
    m = report(val, pred)
    print(f"rows: train {n_tr}, validation {n_va}, trees {booster.num_boosted_rounds()}")
    print(f"validation MAE: all {m['validation_mae']:.1f} days, "
          f"parts failing within 60 days {m['near_failure_mae']:.1f} days")

    # The Lambda runs the exported JSON in pure Python: it must agree with xgboost itself.
    tree = TreeModel(booster.save_raw("json"))
    sample = val[:: max(1, len(val) // 500)]
    diff = max(abs(tree.predict(list(r[1:])) - p)
               for r, p in zip(sample, booster.predict(xgb.DMatrix(sample[:, 1:]))))
    print(f"pure-Python vs xgboost max difference: {diff:.2e} days")
    return booster


def run_demo_fleet(booster, seed):
    """-> {(vehicle, component): {"trend": first warning day, "model": first warning day, "limit": day}}"""
    out = {}
    for v in make_fleet(20, seed):
        history = {c: [] for c in COMPONENTS}
        for day in range(DAYS):
            vals = extract(v.telemetry(day))
            feats, comps = [], []
            for comp, value in vals.items():
                history[comp] = (history[comp] + [(day, value)])[-WINDOW:]
                f = features(comp, history[comp])
                if f is not None:
                    feats.append(f)
                    comps.append(comp)
            preds = dict(zip(comps, booster.predict(xgb.DMatrix(np.array(feats)))) if feats else {})
            for comp, value in vals.items():
                rec = out.setdefault((v.vehicle_id, comp), {"trend": None, "model": None, "limit": None})
                for key, model_rul in (("trend", None), ("model", preds.get(comp))):
                    if key == "model" and model_rul is None:
                        continue
                    status = score(comp, history[comp], model_rul=None if model_rul is None else float(model_rul))
                    if status["status"] != "HEALTHY" and rec[key] is None:
                        rec[key] = day
                if crossed(comp, value) and rec["limit"] is None:
                    rec["limit"] = day
    return out


def main():
    booster = train_local(ROOT / "ml" / ".data")
    faulty = {(vid, FAULT_COMPONENT[f]) for vid, (f, _) in FAULTS.items()}

    res = run_demo_fleet(booster, 42)
    print("\nDemo fleet (seed 42)            trend warns  model warns  limit day")
    for vid, (fault, _) in sorted(FAULTS.items()):
        r = res[(vid, FAULT_COMPONENT[fault])]
        print(f"  {vid} {fault:20s}  day {r['trend']!s:>4}     day {r['model']!s:>4}     day {r['limit']}")

    seeds = range(1, 21)
    totals = {"trend": [0, 0], "model": [0, 0]}  # [false alarms, faults missed or <14 days warning]
    lead = {"trend": [], "model": []}
    for seed in seeds:
        res = run_demo_fleet(booster, seed)
        for key, rec in res.items():
            for src in ("trend", "model"):
                if key not in faulty and rec[src] is not None:
                    totals[src][0] += 1
                if key in faulty:
                    if rec[src] is None or (rec["limit"] is not None and rec["limit"] - rec[src] < 14):
                        totals[src][1] += 1
                    elif rec["limit"] is not None:
                        lead[src].append(rec["limit"] - rec[src])
    print(f"\nAcross {len(seeds)} demo fleets ({len(seeds) * 16} healthy vehicles, {len(seeds) * 4} faults):")
    for src in ("trend", "model"):
        print(f"  {src:5s}  false alarms {totals[src][0]:3d}   missed/late {totals[src][1]:3d}   "
              f"mean lead time {np.mean(lead[src]):.0f} days")


if __name__ == "__main__":
    main()
