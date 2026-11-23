"""EXTRACT: pull raw weather, track and lap data into DataFrames.

Sources
-------
* Weather  - Open-Meteo forecast API (free, no key). Falls back to a deterministic
             synthetic forecast when offline so the dashboard always works.
* Track    - static circuit catalogue in ``f1cc.config``.
* Laps     - either a simulated race session (physics-flavoured generator) or a
             user-supplied CSV that follows ``LAP_SCHEMA``.
"""
from __future__ import annotations

import io
import json
import ssl
import urllib.parse
import urllib.request
import zlib
from datetime import date, datetime

import numpy as np
import pandas as pd

from f1cc.config import (
    CIRCUITS,
    CLIFF_PENALTY_S_PER_LAP2,
    COMPOUNDS,
    DRIVERS,
    FUEL_EFFECT_S_PER_KG,
    FUEL_START_KG,
    LAP_SCHEMA,
    TRACK_TEMP_EFFECT_S_PER_C,
    TRACK_TEMP_REF_C,
    Circuit,
)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_FIELDS = [
    "temperature_2m", "relative_humidity_2m", "precipitation_probability",
    "precipitation", "wind_speed_10m", "wind_direction_10m", "shortwave_radiation",
]


class ExtractError(RuntimeError):
    pass


# --------------------------------------------------------------------------- weather
def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi  # shipped alongside pip/requests in most environments

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # pragma: no cover - depends on environment
        return ssl.create_default_context()


def fetch_weather_live(circuit: Circuit, days: int = 3, timeout: float = 8.0) -> pd.DataFrame:
    """Hourly forecast for the circuit location, in the circuit's local time."""
    params = {
        "latitude": circuit.lat,
        "longitude": circuit.lon,
        "hourly": ",".join(HOURLY_FIELDS),
        "forecast_days": days,
        "timezone": "auto",
        "wind_speed_unit": "kmh",
    }
    url = f"{OPEN_METEO_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "f1-command-centre/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise ExtractError(f"Open-Meteo request failed: {exc}") from exc

    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise ExtractError("Open-Meteo response missing 'hourly' block")
    df = pd.DataFrame(hourly)
    df.attrs["source"] = "open-meteo"
    df.attrs["timezone"] = payload.get("timezone", "local")
    return df


def synthetic_weather(circuit: Circuit, days: int = 3, seed: int | None = None) -> pd.DataFrame:
    """Plausible diurnal forecast, deterministic per circuit and day."""
    today = date.today()
    if seed is None:
        seed = zlib.crc32(f"{circuit.key}-{today.isoformat()}".encode())
    rng = np.random.default_rng(seed)
    hours = pd.date_range(datetime(today.year, today.month, today.day), periods=24 * days, freq="h")
    h = hours.hour.to_numpy()
    idx = np.arange(len(h))

    mean_temp = 31.0 - 0.28 * abs(circuit.lat) + rng.normal(0, 1.5)
    air = mean_temp + 5.5 * np.sin((h - 9) / 24 * 2 * np.pi) + rng.normal(0, 0.4, len(h))
    radiation = np.clip(850 * np.sin(np.pi * (h - 6) / 12), 0, None) * rng.uniform(0.6, 1.0)

    # A weather front with a random arrival time gives the strategy engine something to react to.
    front_centre = rng.uniform(10, 24 * days - 6)
    front_strength = rng.choice([0.0, 0.0, 35.0, 70.0])
    rain_prob = np.clip(
        8 + front_strength * np.exp(-((idx - front_centre) ** 2) / 18) + rng.normal(0, 3, len(h)), 0, 100
    )
    precip = np.where(rain_prob > 55, rng.gamma(1.5, 0.6, len(h)), 0.0)
    radiation = radiation * (1 - rain_prob / 150)

    df = pd.DataFrame({
        "time": hours.strftime("%Y-%m-%dT%H:%M"),
        "temperature_2m": air.round(1),
        "relative_humidity_2m": np.clip(55 - 1.2 * (air - mean_temp) + rain_prob * 0.3, 15, 100).round(0),
        "precipitation_probability": rain_prob.round(0),
        "precipitation": precip.round(1),
        "wind_speed_10m": np.clip(rng.normal(14, 4, len(h)), 0, None).round(1),
        "wind_direction_10m": ((rng.uniform(0, 360) + rng.normal(0, 15, len(h))) % 360).round(0),
        "shortwave_radiation": radiation.round(0),
    })
    df.attrs["source"] = "synthetic"
    df.attrs["timezone"] = "local (simulated)"
    return df


def extract_weather(circuit: Circuit, live: bool = True, days: int = 3) -> tuple[pd.DataFrame, str]:
    """Try live data first; fall back to the synthetic generator. Returns (df, note)."""
    if live:
        try:
            df = fetch_weather_live(circuit, days=days)
            return df, f"Live forecast from Open-Meteo ({df.attrs['timezone']})"
        except ExtractError as exc:
            note = f"Live weather unavailable, using synthetic forecast. Reason: {exc}"
            return synthetic_weather(circuit, days=days), note
    return synthetic_weather(circuit, days=days), "Synthetic forecast (offline mode)"


# --------------------------------------------------------------------------- laps
def _strategy_for(rng: np.random.Generator, laps: int) -> list[tuple[str, int]]:
    """Pick a plausible tyre plan: list of (compound, stint_length)."""
    plans = [
        (["MEDIUM", "HARD"], [0.42]),
        (["SOFT", "HARD"], [0.30]),
        (["HARD", "MEDIUM"], [0.58]),
        (["SOFT", "MEDIUM", "HARD"], [0.25, 0.55]),
        (["MEDIUM", "HARD", "SOFT"], [0.35, 0.78]),
        (["MEDIUM", "MEDIUM", "HARD"], [0.30, 0.62]),
    ]
    compounds, splits = plans[rng.integers(len(plans))]
    cut_laps = [int(round(laps * (s + rng.normal(0, 0.04)))) for s in splits]
    cut_laps = sorted(max(4, min(laps - 4, c)) for c in cut_laps)
    bounds = [0, *cut_laps, laps]
    return [(c, bounds[i + 1] - bounds[i]) for i, c in enumerate(compounds) if bounds[i + 1] > bounds[i]]


def simulate_race_laps(
    circuit: Circuit,
    track_temp_by_lap: np.ndarray | None = None,
    seed: int = 7,
    race_laps: int | None = None,
) -> pd.DataFrame:
    """Generate a race lap chart that behaves like real timing data.

    The generator uses the same physical effects the analytics model later has to
    recover (fuel burn, compound offset, non-linear degradation, track temperature,
    driver pace), plus the messy parts: lap-1 traffic, in/out laps, a possible
    safety car and random driver errors.
    """
    rng = np.random.default_rng(seed)
    laps = race_laps or circuit.laps
    if track_temp_by_lap is None:
        track_temp_by_lap = np.full(laps, TRACK_TEMP_REF_C)
    track_temp_by_lap = np.resize(np.asarray(track_temp_by_lap, dtype=float), laps)

    burn = FUEL_START_KG / laps
    sc_start = int(rng.integers(laps // 4, 3 * laps // 4)) if rng.random() < 0.45 else None
    sc_laps = set(range(sc_start, sc_start + int(rng.integers(3, 6)))) if sc_start else set()

    rows = []
    for d_idx, (code, team, _) in enumerate(DRIVERS):
        pace = rng.normal(0, 0.30) + 0.06 * d_idx      # rough grid order
        tyre_mgmt = rng.uniform(0.85, 1.15)             # <1 = gentle on tyres
        plan = _strategy_for(rng, laps)
        lap_no = 0
        for stint_no, (comp_name, stint_len) in enumerate(plan, start=1):
            comp = COMPOUNDS[comp_name]
            for age in range(1, stint_len + 1):
                lap_no += 1
                fuel = FUEL_START_KG - burn * (lap_no - 1)
                temp = track_temp_by_lap[lap_no - 1]
                deg_scale = circuit.abrasiveness * tyre_mgmt * (1 + 0.012 * (temp - TRACK_TEMP_REF_C))
                deg = deg_scale * (comp.deg_lin * age + comp.deg_quad * age**2)
                if age > comp.life_laps:
                    deg += CLIFF_PENALTY_S_PER_LAP2 * 0.1 * (age - comp.life_laps) ** 2
                t = (
                    circuit.base_lap_s
                    + comp.pace_offset_s
                    + deg
                    + FUEL_EFFECT_S_PER_KG * fuel
                    + TRACK_TEMP_EFFECT_S_PER_C * (temp - TRACK_TEMP_REF_C)
                    + pace
                    + rng.normal(0, 0.18)
                )
                status = "GREEN"
                pit_out = age == 1 and stint_no > 1
                pit_in = age == stint_len and lap_no < laps
                if lap_no == 1:
                    t += 3.5 + rng.uniform(0, 2.5)
                if pit_out:
                    t += comp.warmup_s + circuit.pit_loss_s * 0.45
                if pit_in:
                    t += circuit.pit_loss_s * 0.55
                if lap_no in sc_laps:
                    t *= 1.32
                    status = "SC"
                if rng.random() < 0.015:   # lock-up / track limits
                    t += rng.uniform(0.8, 2.5)
                rows.append({
                    "driver": code, "team": team, "lap": lap_no, "lap_time_s": round(float(t), 3),
                    "compound": comp_name, "tyre_age": age, "stint": stint_no,
                    "pit_in": bool(pit_in), "pit_out": bool(pit_out), "track_status": status,
                })
    df = pd.DataFrame(rows, columns=LAP_SCHEMA)
    df.attrs["source"] = "simulated"
    return df


def read_laps_csv(data: bytes | str | io.IOBase) -> pd.DataFrame:
    """Load user-supplied timing data and validate it against LAP_SCHEMA."""
    buf = io.BytesIO(data) if isinstance(data, bytes) else data
    df = pd.read_csv(buf)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in ("driver", "lap", "lap_time_s", "compound", "tyre_age") if c not in df.columns]
    if missing:
        raise ExtractError(f"CSV missing required columns: {', '.join(missing)}")
    defaults = {"team": "Unknown", "stint": 1, "pit_in": False, "pit_out": False, "track_status": "GREEN"}
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
    df["compound"] = df["compound"].astype(str).str.upper().str.strip()
    unknown = sorted(set(df["compound"]) - set(COMPOUNDS))
    if unknown:
        raise ExtractError(f"Unsupported compounds in CSV: {', '.join(unknown)} (use SOFT/MEDIUM/HARD)")
    out = df[LAP_SCHEMA].copy()
    out.attrs["source"] = "csv upload"
    return out


def get_circuit(key: str) -> Circuit:
    try:
        return CIRCUITS[key]
    except KeyError as exc:
        raise ExtractError(f"Unknown circuit '{key}'") from exc
