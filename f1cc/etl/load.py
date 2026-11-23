"""LOAD: persist curated tables to a local SQLite warehouse for history and auditing."""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import pandas as pd

DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "warehouse.db"


def load_to_sqlite(laps: pd.DataFrame, weather: pd.DataFrame, circuit_key: str, report: dict,
                   db_path: Path | str = DEFAULT_DB) -> str:
    """Write curated laps & weather, tagged with a run id. Returns the run id."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now().replace(microsecond=0)
    run_id = f"{circuit_key}-{started:%Y%m%d%H%M%S}"

    laps_out = laps.assign(run_id=run_id)
    weather_out = weather.assign(run_id=run_id, time=weather["time"].dt.strftime("%Y-%m-%dT%H:%M"))
    with closing(sqlite3.connect(db_path)) as con:
        laps_out.to_sql("laps", con, if_exists="append", index=False)
        weather_out.to_sql("weather", con, if_exists="append", index=False)
        con.execute(
            "CREATE TABLE IF NOT EXISTS pipeline_runs (run_id TEXT PRIMARY KEY, started_at TEXT, circuit TEXT, "
            "rows_laps INTEGER, rows_weather INTEGER, notes TEXT)"
        )
        con.execute(
            "INSERT OR REPLACE INTO pipeline_runs VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, started.isoformat(), circuit_key, len(laps_out), len(weather_out), json.dumps(report)),
        )
        con.commit()
    return run_id


def recent_runs(db_path: Path | str = DEFAULT_DB, limit: int = 10) -> pd.DataFrame:
    db_path = Path(db_path)
    if not db_path.exists():
        return pd.DataFrame(columns=["run_id", "started_at", "circuit", "rows_laps", "rows_weather"])
    with closing(sqlite3.connect(db_path)) as con:
        try:
            return pd.read_sql(
                "SELECT run_id, started_at, circuit, rows_laps, rows_weather FROM pipeline_runs "
                "ORDER BY started_at DESC LIMIT ?", con, params=(limit,),
            )
        except Exception:
            return pd.DataFrame()
