"""Orchestrates Extract -> Transform -> Load and records a step log for the UI."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from f1cc.config import Circuit
from f1cc.etl import extract, load, transform


@dataclass
class PipelineResult:
    circuit: Circuit
    race_start: datetime
    weather: pd.DataFrame          # full forecast
    race_weather: pd.DataFrame     # forecast rows covering the race
    laps: pd.DataFrame             # curated lap table
    report: dict
    weather_note: str
    run_id: str | None
    log: list[dict] = field(default_factory=list)


def run_pipeline(
    circuit_key: str,
    race_start: datetime,
    live_weather: bool = True,
    seed: int = 7,
    laps_csv: bytes | None = None,
    persist: bool = True,
    db_path: Path | str | None = None,
) -> PipelineResult:
    log: list[dict] = []

    def step(name: str, fn, detail_fn=lambda r: ""):
        t0 = time.perf_counter()
        result = fn()
        log.append({"step": name, "ms": round((time.perf_counter() - t0) * 1000, 1), "detail": detail_fn(result)})
        return result

    circuit = step("extract: circuit", lambda: extract.get_circuit(circuit_key), lambda c: c.name)
    raw_weather, note = step(
        "extract: weather", lambda: extract.extract_weather(circuit, live=live_weather),
        lambda r: f"{len(r[0])} hourly rows - {r[0].attrs.get('source')}",
    )
    weather = step("transform: weather", lambda: transform.transform_weather(raw_weather),
                   lambda w: f"track temp {w['track_temp_c'].min():.0f}-{w['track_temp_c'].max():.0f} C")
    window = transform.race_weather(weather, race_start, circuit)
    temps = transform.track_temp_per_lap(weather, race_start, circuit)

    if laps_csv is not None:
        raw_laps = step("extract: laps (csv)", lambda: extract.read_laps_csv(laps_csv), lambda d: f"{len(d)} rows")
    else:
        raw_laps = step("extract: laps (simulated)",
                        lambda: extract.simulate_race_laps(circuit, track_temp_by_lap=temps, seed=seed),
                        lambda d: f"{len(d)} rows, {d['driver'].nunique()} drivers")

    laps, report = step("transform: laps", lambda: transform.transform_laps(raw_laps, circuit, temps),
                        lambda r: f"{r[1]['clean_laps']} clean / {r[1]['rows_valid']} valid laps")

    run_id = None
    if persist:
        kwargs = {"db_path": db_path} if db_path else {}
        run_id = step("load: sqlite", lambda: load.load_to_sqlite(laps, weather, circuit.key, report, **kwargs),
                      lambda rid: rid)

    return PipelineResult(circuit, race_start, weather, window, laps, report, note, run_id, log)
