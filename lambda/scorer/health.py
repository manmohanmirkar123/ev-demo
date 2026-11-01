"""Component health scoring: anomaly vs. fleet baseline + trend-based remaining useful life (RUL).

Pure Python (no third-party deps) so the same code runs in Lambda, the tests and the dashboard.
"""

# bad: which direction is degradation. wear: the indicator declines for every vehicle by design
# (e.g. brake pads), so deviation from the fleet mean is not an anomaly — only RUL matters.
COMPONENTS = {
    "motor_bearing": {"label": "Motor bearing", "unit": "g RMS", "bad": "up", "threshold": 1.2,
                      "mean": 0.30, "std": 0.04, "wear": False},
    "inverter_thermal": {"label": "Inverter cooling", "unit": "°C over coolant", "bad": "up", "threshold": 35.0,
                         "mean": 15.0, "std": 1.0, "wear": False},
    "coolant_pump": {"label": "Coolant pump", "unit": "L/min", "bad": "down", "threshold": 6.0,
                     "mean": 12.0, "std": 0.4, "wear": False},
    "brake_pads": {"label": "Brake pads", "unit": "mm (thinnest)", "bad": "down", "threshold": 3.0,
                   "mean": None, "std": None, "wear": True},
    "tyres": {"label": "Tyres", "unit": "psi (lowest)", "bad": "down", "threshold": 30.0,
              "mean": 36.0, "std": 0.5, "wear": False},
}

SEVERITY = {"HEALTHY": 0, "AT_RISK": 1, "CRITICAL": 2}

TREND_POINTS = 15     # days used for the slope fit
MIN_POINTS = 8        # no RUL until we have this many days
SMOOTH_POINTS = 5     # current value = mean of the last N days
Z_TREND_GATE = 2.0    # non-wear parts: only trust a trend once the part visibly deviates
Z_AT_RISK = 4.0
RUL_AT_RISK = 45.0
RUL_CRITICAL = 14.0


def extract(msg):
    """Telemetry message -> {component: health indicator value}."""
    tyres = msg["tyres"]
    brakes = msg["brakes"]
    return {
        "motor_bearing": msg["motor"]["vibration_g"],
        "inverter_thermal": msg["inverter"]["temp_c"] - msg["cooling"]["coolant_temp_c"],
        "coolant_pump": msg["cooling"]["pump_flow_lpm"],
        "brake_pads": min(brakes["pad_mm_front"], brakes["pad_mm_rear"]),
        "tyres": min(tyres["fl"], tyres["fr"], tyres["rl"], tyres["rr"]),
    }


# ---------- ML features (shared by ml/ training and the Lambda, so they can't drift apart) ----------

# Typical healthy-to-limit distance per component; puts every component's features on one scale.
SCALE = {"motor_bearing": 0.9, "inverter_thermal": 20.0, "coolant_pump": 6.0, "brake_pads": 9.0, "tyres": 6.0}
RUL_CAP = 120.0  # model horizon in days: "120" means "not within the next 120 days"
FEATURES = [f"is_{c}" for c in COMPONENTS] + [
    "margin", "slope_15", "slope_5", "accel", "noise", "z", "trend_rul",
]


def features(component, history, baseline=None):
    """Window of (day, value) -> feature vector (FEATURES order), or None if too little history."""
    if len(history) < MIN_POINTS:
        return None
    spec = {**COMPONENTS[component], **(baseline or {})}
    sign = 1 if spec["bad"] == "up" else -1
    scale = SCALE[component]
    window = history[-TREND_POINTS:]
    recent = [v for _, v in history[-SMOOTH_POINTS:]]
    current = sum(recent) / len(recent)

    margin = sign * (spec["threshold"] - current) / scale          # 1 ≈ healthy, 0 = at the limit
    slope_15 = sign * _slope(window) / scale                        # >0 = degrading
    slope_5 = sign * _slope(history[-SMOOTH_POINTS:]) / scale
    mean = sum(v for _, v in window) / len(window)
    noise = (sum((v - mean) ** 2 for _, v in window) / len(window)) ** 0.5 / scale
    z = 0.0 if spec["wear"] or not spec["std"] else sign * (current - spec["mean"]) / spec["std"]
    trend_rul = min(RUL_CAP * 3, margin / slope_15) if slope_15 > 0 else RUL_CAP * 3
    trend_rul = max(trend_rul, 0.0)

    onehot = [1.0 if c == component else 0.0 for c in COMPONENTS]
    return onehot + [margin, slope_15, slope_5, slope_5 - slope_15, noise, z, trend_rul]


def _slope(points):
    n = len(points)
    mx = sum(d for d, _ in points) / n
    my = sum(v for _, v in points) / n
    sxx = sum((d - mx) ** 2 for d, _ in points)
    if sxx == 0:
        return 0.0
    return sum((d - mx) * (v - my) for d, v in points) / sxx


def score(component, history, baseline=None, model_rul=None):
    """history: [(sim_day, value), ...] oldest first. baseline: optional {"mean", "std"} learned from the fleet.

    model_rul: days-to-limit from the SageMaker model. When given it replaces the linear trend for the
    status decision; the trend is still reported alongside for comparison.
    """
    spec = {**COMPONENTS[component], **(baseline or {})}
    sign = 1 if spec["bad"] == "up" else -1
    recent = [v for _, v in history[-SMOOTH_POINTS:]]
    current = sum(recent) / len(recent)

    z = None
    if not spec["wear"] and spec["std"]:
        z = sign * (current - spec["mean"]) / spec["std"]

    slope = None
    rul = None
    if len(history) >= MIN_POINTS:
        slope = _slope(history[-TREND_POINTS:])
        degrading = sign * slope > 0
        gate_open = spec["wear"] or (z is not None and z >= Z_TREND_GATE)
        if degrading and gate_open:
            rul = max(0.0, sign * (spec["threshold"] - current) / (sign * slope))
    trend_rul = rul

    if model_rul is not None:
        # Same gate as the trend: the anomaly score decides *whether* a part is degrading,
        # the model estimates *how soon* it reaches the limit.
        gate_open = spec["wear"] or (z is not None and z >= Z_TREND_GATE)
        rul = None if (model_rul >= RUL_CAP * 0.9 or not gate_open) else max(0.0, model_rul)

    past_threshold = sign * (current - spec["threshold"]) >= 0
    if past_threshold or (rul is not None and rul <= RUL_CRITICAL):
        status = "CRITICAL"
    elif (rul is not None and rul <= RUL_AT_RISK) or (z is not None and z >= Z_AT_RISK):
        status = "AT_RISK"
    else:
        status = "HEALTHY"

    return {
        "value": round(current, 3),
        "z": None if z is None else round(z, 2),
        "slope_per_day": None if slope is None else round(slope, 5),
        "rul_days": None if rul is None else round(rul, 1),
        "rul_trend_days": None if trend_rul is None else round(trend_rul, 1),
        "rul_model_days": None if model_rul is None else round(model_rul, 1),
        "rul_source": "model" if model_rul is not None else "trend",
        "status": status,
    }
