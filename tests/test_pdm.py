"""Offline end-to-end check: simulated fleet -> scoring engine, no AWS needed.

Every injected fault must be flagged before its DTC fires, and healthy vehicles must stay quiet.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "lambda" / "scorer"), str(ROOT / "simulator")]

from fleet import FAULTS, make_fleet  # noqa: E402
from health import COMPONENTS, extract, score  # noqa: E402

DAYS = 120
WINDOW = 30
FAULT_COMPONENT = {
    "motor_bearing": "motor_bearing",
    "coolant_pump": "coolant_pump",
    "tyre_leak": "tyres",
    "aggressive_braking": "brake_pads",
}


def run_fleet():
    """-> {(vehicle, component): {"first_warning": day, "threshold_day": day}}"""
    out = {}
    for v in make_fleet(20):
        history = {c: [] for c in COMPONENTS}
        for day in range(DAYS):
            msg = v.telemetry(day)
            for comp, value in extract(msg).items():
                history[comp] = (history[comp] + [(day, value)])[-WINDOW:]
                result = score(comp, history[comp])
                rec = out.setdefault((v.vehicle_id, comp), {"first_warning": None, "threshold_day": None})
                if result["status"] != "HEALTHY" and rec["first_warning"] is None:
                    rec["first_warning"] = day
                spec = COMPONENTS[comp]
                crossed = value >= spec["threshold"] if spec["bad"] == "up" else value <= spec["threshold"]
                if crossed and rec["threshold_day"] is None:
                    rec["threshold_day"] = day
    return out


RESULTS = run_fleet()


def test_every_fault_is_warned_before_threshold():
    for vid, (fault, _) in FAULTS.items():
        rec = RESULTS[(vid, FAULT_COMPONENT[fault])]
        assert rec["first_warning"] is not None, f"{vid} {fault} never flagged"
        if rec["threshold_day"] is not None:
            lead = rec["threshold_day"] - rec["first_warning"]
            assert lead >= 14, f"{vid} {fault}: only {lead} days warning"


def test_no_false_alarms_on_healthy_parts():
    faulty = {(vid, FAULT_COMPONENT[f]) for vid, (f, _) in FAULTS.items()}
    # A failing coolant pump legitimately heats the inverter too.
    faulty.add(("EV-007", "inverter_thermal"))
    false_alarms = [k for k, rec in RESULTS.items() if k not in faulty and rec["first_warning"] is not None]
    assert not false_alarms, f"false alarms: {false_alarms}"


if __name__ == "__main__":
    for (vid, comp), rec in sorted(RESULTS.items()):
        if rec["first_warning"] is not None or rec["threshold_day"] is not None:
            print(f"{vid} {comp:17s} warned day {rec['first_warning']}  threshold crossed day {rec['threshold_day']}")
