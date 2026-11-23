import sqlite3

import pandas as pd
import pytest

from f1cc.config import CIRCUITS, LAP_SCHEMA
from f1cc.etl import extract, transform
from f1cc.etl.load import load_to_sqlite, recent_runs
from f1cc.etl.pipeline import run_pipeline


def test_synthetic_weather_is_deterministic():
    c = CIRCUITS["monza"]
    a = extract.synthetic_weather(c, seed=1)
    b = extract.synthetic_weather(c, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 72
    assert set(extract.HOURLY_FIELDS) <= set(a.columns)


def test_live_weather_falls_back_when_offline(monkeypatch):
    def boom(*_a, **_k):
        raise extract.ExtractError("no network")

    monkeypatch.setattr(extract, "fetch_weather_live", boom)
    df, note = extract.extract_weather(CIRCUITS["spa"], live=True)
    assert df.attrs["source"] == "synthetic"
    assert "unavailable" in note


def test_transform_weather_adds_track_temp():
    w = transform.transform_weather(extract.synthetic_weather(CIRCUITS["bahrain"], seed=3))
    assert {"track_temp_c", "condition"} <= set(w.columns)
    assert (w["track_temp_c"] >= w["air_temp_c"]).all()
    assert pd.api.types.is_datetime64_any_dtype(w["time"])


def test_simulated_laps_schema_and_distance():
    c = CIRCUITS["silverstone"]
    laps = extract.simulate_race_laps(c, seed=3)
    assert list(laps.columns) == LAP_SCHEMA
    assert (laps.groupby("driver")["lap"].max() == c.laps).all()


def test_transform_laps_flags_and_report(result):
    laps, rep = result.laps, result.report
    assert rep["clean_laps"] > 0.7 * rep["rows_valid"]
    assert not laps.loc[laps["pit_in"] | laps["pit_out"], "is_clean"].any()
    assert laps.groupby("lap")["position"].apply(lambda p: sorted(p) == list(range(1, len(p) + 1))).all()
    assert laps["fuel_kg"].between(0, 100).all()


def test_csv_round_trip(result, race_start, tmp_path):
    csv = result.laps[LAP_SCHEMA].to_csv(index=False).encode()
    r2 = run_pipeline("silverstone", race_start, live_weather=False, laps_csv=csv, persist=False)
    assert r2.report["rows_valid"] == result.report["rows_valid"]


def test_csv_validation_errors():
    with pytest.raises(extract.ExtractError, match="missing"):
        extract.read_laps_csv(b"driver,lap\nAAA,1\n")
    with pytest.raises(extract.ExtractError, match="Unsupported"):
        extract.read_laps_csv(b"driver,lap,lap_time_s,compound,tyre_age\nAAA,1,90.1,WET,1\n")


def test_load_to_sqlite(result, tmp_path):
    db = tmp_path / "wh.db"
    run_id = load_to_sqlite(result.laps, result.weather, "silverstone", result.report, db_path=db)
    with sqlite3.connect(db) as con:
        n = con.execute("SELECT COUNT(*) FROM laps WHERE run_id = ?", (run_id,)).fetchone()[0]
    assert n == len(result.laps)
    assert recent_runs(db).iloc[0]["run_id"] == run_id
