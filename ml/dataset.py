"""Run-to-failure training data from simulated fleet history.

Each row is one (vehicle, component, day) window: label = days until that component actually crossed its limit,
capped at RUL_CAP. In a real OEM this label comes from repair / warranty records joined to telemetry.
Format is what SageMaker's built-in XGBoost expects: CSV, label first, no header.
"""
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "lambda" / "scorer"), str(ROOT / "simulator")]

from fleet import make_history_fleet  # noqa: E402
from health import COMPONENTS, MIN_POINTS, RUL_CAP, features, extract  # noqa: E402

HORIZON = 300       # simulated days per history vehicle
WINDOW = 30         # same history window as the Lambda
KEEP_CAPPED = 0.15  # "nothing failing soon" rows dominate; keep a sample so the model still learns them


def crossed(component, value):
    spec = COMPONENTS[component]
    return value >= spec["threshold"] if spec["bad"] == "up" else value <= spec["threshold"]


def build(n_vehicles, seed):
    rng = random.Random(seed)
    rows = []
    for v in make_history_fleet(n_vehicles, seed):
        series = {c: [] for c in COMPONENTS}
        for day in range(HORIZON):
            for comp, value in extract(v.telemetry(day)).items():
                series[comp].append((day, value))

        for comp, points in series.items():
            fail_day = next((d for d, val in points if crossed(comp, val)), None)
            for i in range(MIN_POINTS - 1, fail_day if fail_day is not None else HORIZON):
                day = points[i][0]
                if fail_day is None:
                    if HORIZON - day < RUL_CAP:
                        break  # censored: we don't know whether it fails within the horizon
                    label = RUL_CAP
                else:
                    label = min(RUL_CAP, fail_day - day)
                if label >= RUL_CAP and rng.random() > KEEP_CAPPED:
                    continue
                rows.append([label] + features(comp, points[max(0, i - WINDOW + 1):i + 1]))
    return rows


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(",".join(f"{x:.6g}" for x in r) + "\n")
    return path


def build_splits(out_dir, train_vehicles=400, val_vehicles=100):
    """Train and validation come from different fleets (different seeds), so no vehicle is in both."""
    train = build(train_vehicles, seed=1)
    val = build(val_vehicles, seed=2)
    return (write_csv(train, out_dir / "train" / "train.csv"),
            write_csv(val, out_dir / "validation" / "validation.csv"),
            len(train), len(val))


if __name__ == "__main__":
    out = ROOT / "ml" / ".data"
    tr, va, n_tr, n_va = build_splits(out)
    print(f"train: {n_tr} rows -> {tr}\nvalidation: {n_va} rows -> {va}")
