"""TRANSFORM: clean, enrich and join weather + lap data into analysis-ready tables."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from f1cc.config import FUEL_START_KG, Circuit

WEATHER_RENAME = {
    "temperature_2m": "air_temp_c",
    "relative_humidity_2m": "humidity_pct",
    "precipitation_probability": "rain_prob_pct",
    "precipitation": "precip_mm",
    "wind_speed_10m": "wind_kph",
    "wind_direction_10m": "wind_dir_deg",
    "shortwave_radiation": "radiation_wm2",
}


def estimate_track_temp(air_c: pd.Series, radiation_wm2: pd.Series) -> pd.Series:
    """Empirical asphalt temperature: ambient + solar loading.

    Dark asphalt typically runs 10-20 C above air in full sun; this linear proxy
    (~+19 C at 850 W/m2) is good enough for strategy-level decisions.
    """
    return (air_c + 2.0 + 0.0205 * radiation_wm2.fillna(0)).round(1)


def transform_weather(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=WEATHER_RENAME).copy()
    df["time"] = pd.to_datetime(df["time"])
    for col in WEATHER_RENAME.values():
        if col not in df.columns:
            df[col] = np.nan
    df["rain_prob_pct"] = df["rain_prob_pct"].fillna(0)
    df["precip_mm"] = df["precip_mm"].fillna(0)
    df["track_temp_c"] = estimate_track_temp(df["air_temp_c"], df["radiation_wm2"])
    df["condition"] = np.select(
        [df["precip_mm"] >= 2.5, df["precip_mm"] >= 0.3, df["rain_prob_pct"] >= 40],
        ["WET", "DAMP", "RAIN RISK"],
        default="DRY",
    )
    cols = ["time", "air_temp_c", "track_temp_c", "humidity_pct", "rain_prob_pct",
            "precip_mm", "wind_kph", "wind_dir_deg", "radiation_wm2", "condition"]
    return df[cols].sort_values("time").reset_index(drop=True)


def race_weather(weather: pd.DataFrame, start: datetime, circuit: Circuit) -> pd.DataFrame:
    """Rows of the forecast that cover the race (plus one hour either side)."""
    end = start + timedelta(seconds=circuit.laps * circuit.base_lap_s * 1.08)
    mask = (weather["time"] >= start - timedelta(hours=1)) & (weather["time"] <= end + timedelta(hours=1))
    window = weather.loc[mask]
    return window if not window.empty else weather.head(3)


def track_temp_per_lap(weather: pd.DataFrame, start: datetime, circuit: Circuit, laps: int | None = None) -> np.ndarray:
    """Interpolate forecast track temperature onto each race lap."""
    laps = laps or circuit.laps
    lap_times = [start + timedelta(seconds=i * circuit.base_lap_s) for i in range(laps)]
    # Both sides are naive local timestamps, so compare them on the same epoch basis.
    x = (weather["time"] - pd.Timestamp(0)).dt.total_seconds().to_numpy()
    xp = (pd.to_datetime(pd.Series(lap_times)) - pd.Timestamp(0)).dt.total_seconds().to_numpy()
    return np.interp(xp, x, weather["track_temp_c"].to_numpy())


def transform_laps(laps: pd.DataFrame, circuit: Circuit, track_temp_by_lap: np.ndarray | None = None
                   ) -> tuple[pd.DataFrame, dict]:
    """Type-cast, enrich and flag laps. Returns (laps, data-quality report)."""
    df = laps.copy()
    n_raw = len(df)
    df["lap"] = pd.to_numeric(df["lap"], errors="coerce")
    df["lap_time_s"] = pd.to_numeric(df["lap_time_s"], errors="coerce")
    df["tyre_age"] = pd.to_numeric(df["tyre_age"], errors="coerce")
    df = df.dropna(subset=["lap", "lap_time_s", "tyre_age"])
    df = df[(df["lap_time_s"] > 30) & (df["lap_time_s"] < 400)]
    df = df.drop_duplicates(subset=["driver", "lap"], keep="last")
    df[["lap", "tyre_age", "stint"]] = df[["lap", "tyre_age", "stint"]].astype(int)
    for col in ("pit_in", "pit_out"):
        df[col] = df[col].astype(str).str.lower().isin(["true", "1", "yes"])
    df["track_status"] = df["track_status"].astype(str).str.upper()
    df = df.sort_values(["driver", "lap"]).reset_index(drop=True)

    total_laps = int(df["lap"].max())
    df["fuel_kg"] = (FUEL_START_KG - FUEL_START_KG / total_laps * (df["lap"] - 1)).round(2)
    if track_temp_by_lap is not None:
        temps = np.resize(np.asarray(track_temp_by_lap, dtype=float), total_laps)
        df["track_temp_c"] = temps[df["lap"].to_numpy() - 1].round(1)
    else:
        df["track_temp_c"] = 35.0

    # Race positions from cumulative time
    df["race_time_s"] = df.groupby("driver")["lap_time_s"].cumsum()
    df["position"] = df.groupby("lap")["race_time_s"].rank(method="first").astype(int)
    leader = df.groupby("lap")["race_time_s"].transform("min")
    df["gap_to_leader_s"] = (df["race_time_s"] - leader).round(3)

    # Representative ("clean") laps for modelling
    field_median = df.groupby("lap")["lap_time_s"].transform("median")
    df["is_outlier"] = df["lap_time_s"] > field_median * 1.035
    df["is_clean"] = (
        (df["track_status"] == "GREEN") & ~df["pit_in"] & ~df["pit_out"] & (df["lap"] > 1) & ~df["is_outlier"]
    )

    report = {
        "rows_raw": n_raw,
        "rows_valid": len(df),
        "rows_dropped": n_raw - len(df),
        "clean_laps": int(df["is_clean"].sum()),
        "excluded_sc": int((df["track_status"] != "GREEN").sum()),
        "excluded_pit": int((df["pit_in"] | df["pit_out"]).sum()),
        "excluded_outlier": int(df["is_outlier"].sum()),
        "drivers": int(df["driver"].nunique()),
        "laps": total_laps,
    }
    return df, report
