"""Static reference data: circuits, tyre compounds and model constants."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Circuit:
    key: str
    name: str
    country: str
    lat: float
    lon: float
    laps: int
    length_km: float
    base_lap_s: float      # representative clean-air race lap on fresh softs, zero fuel
    pit_loss_s: float      # time lost driving through the pit lane incl. stop
    abrasiveness: float    # multiplier on tyre degradation (1.0 = average)
    overtaking: float      # 0 (impossible) .. 1 (easy)


CIRCUITS: dict[str, Circuit] = {
    c.key: c
    for c in [
        Circuit("bahrain", "Bahrain International Circuit", "Bahrain", 26.0325, 50.5106, 57, 5.412, 93.5, 22.5, 1.25, 0.75),
        Circuit("jeddah", "Jeddah Corniche Circuit", "Saudi Arabia", 21.6319, 39.1044, 50, 6.174, 89.5, 20.0, 0.85, 0.65),
        Circuit("melbourne", "Albert Park Circuit", "Australia", -37.8497, 144.9680, 58, 5.278, 79.0, 19.5, 0.90, 0.45),
        Circuit("suzuka", "Suzuka Circuit", "Japan", 34.8431, 136.5410, 53, 5.807, 91.5, 22.0, 1.20, 0.40),
        Circuit("monaco", "Circuit de Monaco", "Monaco", 43.7347, 7.4206, 78, 3.337, 73.5, 19.0, 0.55, 0.05),
        Circuit("barcelona", "Circuit de Barcelona-Catalunya", "Spain", 41.5700, 2.2611, 66, 4.657, 76.5, 21.5, 1.15, 0.40),
        Circuit("silverstone", "Silverstone Circuit", "United Kingdom", 52.0786, -1.0169, 52, 5.891, 87.5, 20.5, 1.10, 0.60),
        Circuit("spa", "Circuit de Spa-Francorchamps", "Belgium", 50.4372, 5.9714, 44, 7.004, 105.0, 18.5, 1.00, 0.70),
        Circuit("monza", "Autodromo Nazionale Monza", "Italy", 45.6156, 9.2811, 53, 5.793, 81.5, 24.0, 0.80, 0.75),
        Circuit("singapore", "Marina Bay Street Circuit", "Singapore", 1.2914, 103.8640, 62, 4.940, 94.5, 28.5, 0.95, 0.25),
        Circuit("austin", "Circuit of the Americas", "United States", 30.1328, -97.6411, 56, 5.513, 96.5, 20.0, 1.10, 0.65),
        Circuit("interlagos", "Autódromo José Carlos Pace", "Brazil", -23.7036, -46.6997, 71, 4.309, 72.0, 21.0, 1.00, 0.70),
        Circuit("abudhabi", "Yas Marina Circuit", "Abu Dhabi", 24.4672, 54.6031, 58, 5.281, 86.0, 21.5, 0.90, 0.55),
    ]
}


@dataclass(frozen=True)
class Compound:
    name: str
    color: str
    pace_offset_s: float   # vs SOFT on lap 1 of stint
    deg_lin: float         # s / lap of tyre age
    deg_quad: float        # s / lap^2 of tyre age
    life_laps: int         # laps before the performance cliff
    warmup_s: float        # out-lap penalty on a cold tyre


COMPOUNDS: dict[str, Compound] = {
    "SOFT": Compound("SOFT", "#E8002D", 0.00, 0.085, 0.0016, 20, 0.3),
    "MEDIUM": Compound("MEDIUM", "#FFD12E", 0.55, 0.050, 0.0008, 32, 0.7),
    "HARD": Compound("HARD", "#F0F0EC", 1.05, 0.030, 0.0004, 45, 1.2),
}
DRY_COMPOUNDS = list(COMPOUNDS)

FUEL_START_KG = 100.0
FUEL_EFFECT_S_PER_KG = 0.032
TRACK_TEMP_REF_C = 35.0
TRACK_TEMP_EFFECT_S_PER_C = 0.018
CLIFF_PENALTY_S_PER_LAP2 = 0.35   # added per lap^2 beyond compound life

# Fictional grid: the dashboard is an engineering demo, not tied to a real season.
DRIVERS: list[tuple[str, str, str]] = [
    ("VAL", "Vantage Racing", "#00A3E0"),
    ("KOR", "Vantage Racing", "#00A3E0"),
    ("HAL", "Halcyon GP", "#FF6B00"),
    ("DUB", "Halcyon GP", "#FF6B00"),
    ("RIZ", "Ridgeline Motorsport", "#2ECC71"),
    ("MAR", "Ridgeline Motorsport", "#2ECC71"),
    ("OKA", "Corsa Nova", "#E74C3C"),
    ("LEF", "Corsa Nova", "#E74C3C"),
    ("SAN", "Aurora Racing", "#9B59B6"),
    ("PEL", "Aurora Racing", "#9B59B6"),
]
TEAM_COLORS = {team: color for _, team, color in DRIVERS}

LAP_SCHEMA = [
    "driver", "team", "lap", "lap_time_s", "compound", "tyre_age",
    "stint", "pit_in", "pit_out", "track_status",
]
