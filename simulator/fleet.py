"""Synthetic EV fleet. Each call to telemetry(day) returns one day's summarised battery (BMS) telemetry for a vehicle.

Four vehicles carry an injected battery fault so the demo has something to find; the rest stay healthy.
"""
import math
import random
from datetime import datetime, timezone

# vehicle_id -> (fault, first day the fault starts developing)
FAULTS = {
    "EV-003": ("weak_cell", 20),        # one cell's internal resistance rising: cell voltage spread grows
    "EV-007": ("cooling_plate", 15),    # pack cooling plate clogging: cells run hotter than the coolant
    "EV-011": ("moisture_ingress", 30), # water getting into the pack: HV insulation resistance falls
    "EV-015": ("fast_fade", 10),        # heavy DC fast charging in heat: state of health drops quickly
}


FAULT_TYPES = ("weak_cell", "cooling_plate", "moisture_ingress", "fast_fade")
CELLS_IN_SERIES = 96


class Vehicle:
    def __init__(self, vehicle_id, seed, fault=None, fault_day=0, severity=1.0):
        """fault/fault_day default to the demo FAULTS table; severity scales how fast the fault develops."""
        self.vehicle_id = vehicle_id
        self.rng = random.Random(seed)
        self.fault, self.fault_day = (fault, fault_day) if fault else FAULTS.get(vehicle_id, (None, 0))
        self.severity = severity
        self.odometer_km = self.rng.uniform(5_000, 40_000)
        self.daily_km = self.rng.uniform(35, 80)
        self.soh = self.rng.uniform(90.0, 97.0)       # packs in the fleet are of different ages
        self.insulation_base = self.rng.uniform(2900, 3100)

    def _active(self, fault, day):
        return self.fault == fault and day >= self.fault_day

    def telemetry(self, day):
        g = self.rng.gauss
        km = max(0.0, g(self.daily_km, 8))
        self.odometer_km += km
        t = day - self.fault_day
        s = self.severity

        spread_mv = g(15.0, 1.2)
        if self._active("weak_cell", day):
            spread_mv += 6.0 * (math.exp(0.025 * s * t) - 1)
        cell_v_min = g(3.70, 0.02)
        cell_v_max = cell_v_min + spread_mv / 1000

        coolant = g(28.0, 1.5)
        cell_over_coolant = g(6.0, 0.4)
        if self._active("cooling_plate", day):
            cell_over_coolant += 0.12 * s * t

        insulation = self.insulation_base * g(1.0, 0.025)
        if self._active("moisture_ingress", day):
            insulation *= math.exp(-0.03 * s * t)

        fading = self._active("fast_fade", day)
        self.soh -= (0.25 * s if fading else 0.004) * max(0.0, g(1.0, 0.2))
        dc_fast = sum(self.rng.random() < (0.8 if fading else 0.15) for _ in range(2))

        # DTCs only fire once a limit is crossed — the point of the demo is to warn well before this.
        dtc = []
        if spread_mv > 50:
            dtc.append("BMS_CELL_IMBALANCE")
        if cell_over_coolant > 15:
            dtc.append("BMS_PACK_OVERTEMP")
        if insulation < 500:
            dtc.append("BMS_ISOLATION_FAULT")
        if self.soh < 80:
            dtc.append("BMS_SOH_LOW")

        r = lambda x, n=2: round(x, n)
        return {
            "vehicle_id": self.vehicle_id,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sim_day": day,
            "odometer_km": r(self.odometer_km, 1),
            "battery": {
                "soc_pct": r(self.rng.uniform(20, 95), 1),
                "soh_pct": r(self.soh + g(0, 0.1)),
                "pack_voltage_v": r((cell_v_min + cell_v_max) / 2 * CELLS_IN_SERIES, 1),
                "cell_v_max": r(cell_v_max, 4),
                "cell_v_min": r(cell_v_min, 4),
                "cell_temp_max_c": r(coolant + cell_over_coolant, 1),
                "coolant_temp_c": r(coolant, 1),
                "insulation_kohm": r(insulation, 0),
            },
            "charging": {"dc_fast_sessions": dc_fast, "energy_kwh": r(km * g(0.17, 0.01), 1)},
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
