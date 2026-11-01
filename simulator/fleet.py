"""Synthetic EV fleet. Each call to telemetry(day) returns one day's summarised telemetry for a vehicle.

Four vehicles carry an injected fault so the demo has something to find; the rest stay healthy.
"""
import math
import random
from datetime import datetime, timezone

# vehicle_id -> (fault, first day the fault starts developing)
FAULTS = {
    "EV-003": ("motor_bearing", 20),       # bearing wear: vibration grows exponentially
    "EV-007": ("coolant_pump", 15),        # pump degrading: flow falls, inverter runs hotter
    "EV-011": ("tyre_leak", 30),           # slow puncture front-left
    "EV-015": ("aggressive_braking", 10),  # driver change: pads wear ~8x faster
}


FAULT_TYPES = ("motor_bearing", "coolant_pump", "tyre_leak", "aggressive_braking")


class Vehicle:
    def __init__(self, vehicle_id, seed, fault=None, fault_day=0, severity=1.0):
        """fault/fault_day default to the demo FAULTS table; severity scales how fast the fault develops."""
        self.vehicle_id = vehicle_id
        self.rng = random.Random(seed)
        self.fault, self.fault_day = (fault, fault_day) if fault else FAULTS.get(vehicle_id, (None, 0))
        self.severity = severity
        self.odometer_km = self.rng.uniform(5_000, 40_000)
        self.daily_km = self.rng.uniform(35, 80)
        self.pad_front = self.rng.uniform(7.0, 11.0)
        self.pad_rear = self.pad_front + self.rng.uniform(0.5, 1.5)
        self.tyre_base = {t: self.rng.uniform(35.5, 36.5) for t in ("fl", "fr", "rl", "rr")}

    def _active(self, fault, day):
        return self.fault == fault and day >= self.fault_day

    def telemetry(self, day):
        g = self.rng.gauss
        km = max(0.0, g(self.daily_km, 8))
        self.odometer_km += km
        t = day - self.fault_day
        s = self.severity

        vibration = g(0.30, 0.03)
        if self._active("motor_bearing", day):
            vibration += 0.30 * (math.exp(0.02 * s * t) - 1)

        flow = g(12.0, 0.3)
        if self._active("coolant_pump", day):
            flow -= 0.08 * s * t
        flow = max(flow, 0.5)

        coolant = g(35.0, 1.5) + max(0.0, 12.0 - flow) * 1.2
        inverter = coolant + g(15.0, 0.8) + max(0.0, 12.0 - flow) * 2.0
        motor_temp = g(65.0, 3.0) + max(0.0, vibration - 0.3) * 10

        pad_rate = 0.0016 * s if self._active("aggressive_braking", day) else 0.0002  # mm per km
        self.pad_front = max(0.5, self.pad_front - pad_rate * km)
        self.pad_rear = max(0.5, self.pad_rear - pad_rate * 0.6 * km)

        tyres = {k: g(base, 0.3) for k, base in self.tyre_base.items()}
        if self._active("tyre_leak", day):
            tyres["fl"] -= 0.15 * s * t

        # DTCs only fire once a limit is crossed — the point of the demo is to warn well before this.
        dtc = []
        if vibration > 1.2:
            dtc.append("MOTOR_VIBRATION_HIGH")
        if flow < 6.0:
            dtc.append("COOLANT_FLOW_LOW")
        if min(self.pad_front, self.pad_rear) < 3.0:
            dtc.append("BRAKE_PAD_WEAR_LIMIT")
        if min(tyres.values()) < 30.0:
            dtc.append("TYRE_PRESSURE_LOW")

        r = lambda x, n=2: round(x, n)
        return {
            "vehicle_id": self.vehicle_id,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sim_day": day,
            "odometer_km": r(self.odometer_km, 1),
            "motor": {"temp_c": r(motor_temp, 1), "vibration_g": r(vibration, 3), "rpm": int(g(4200, 300))},
            "inverter": {"temp_c": r(inverter, 1)},
            "cooling": {"coolant_temp_c": r(coolant, 1), "pump_flow_lpm": r(flow)},
            "brakes": {"pad_mm_front": r(self.pad_front, 3), "pad_mm_rear": r(self.pad_rear, 3)},
            "tyres": {k: r(v) for k, v in tyres.items()},
            "dtc": dtc,
        }


def make_fleet(size, seed=42):
    return [Vehicle(f"EV-{i:03d}", seed * 1000 + i) for i in range(1, size + 1)]


def make_history_fleet(size, seed):
    """Training fleet with randomised faults (type, onset, severity); ~30% stay healthy."""
    rng = random.Random(seed)
    fleet = []
    for i in range(size):
        fault = rng.choice(FAULT_TYPES) if rng.random() > 0.3 else None
        fleet.append(Vehicle(f"H-{seed}-{i:04d}", rng.randrange(1 << 30), fault=fault,
                             fault_day=rng.randint(0, 120), severity=rng.uniform(0.5, 2.0)))
    return fleet
