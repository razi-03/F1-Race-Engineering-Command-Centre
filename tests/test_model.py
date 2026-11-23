import numpy as np
import pytest

from f1cc.analytics.lap_model import (
    COMPONENTS,
    LapDeltaModel,
    cumulative_gap,
    delta_decomposition,
    driver_pace_table,
)
from f1cc.config import COMPOUNDS, FUEL_EFFECT_S_PER_KG


def test_model_fits_well(model):
    assert model.r2 > 0.8
    assert model.rmse < 0.4


def test_recovers_physics(model):
    # Fuel effect anchored near the physical prior, compounds ordered by pace.
    assert model.fuel_coef == pytest.approx(FUEL_EFFECT_S_PER_KG, abs=0.01)
    assert model.comp_offset["SOFT"] < model.comp_offset["MEDIUM"] < model.comp_offset["HARD"]
    assert model.tyre_deg("SOFT", 15) > model.tyre_deg("HARD", 15)


def test_driver_pace_is_centred(model):
    assert np.mean(list(model.driver_pace.values())) == pytest.approx(0, abs=1e-9)


def test_prediction_close_to_actual(model, result):
    clean = result.laps[result.laps["is_clean"]]
    err = np.abs(model.predict(clean) - clean["lap_time_s"].to_numpy())
    assert np.median(err) < 0.3


def test_decomposition_sums_to_actual(model, result):
    d = delta_decomposition(model, result.laps, "VAL", "HAL", 20)
    assert list(d["component"][:-1]) == COMPONENTS
    assert d["delta_s"].sum() == pytest.approx(d.attrs["actual"], abs=1e-9)


def test_cumulative_gap_matches_race_time(result):
    g = cumulative_gap(result.laps, "VAL", "KOR")
    last = result.laps[result.laps["lap"] == g["lap"].max()].set_index("driver")["race_time_s"]
    assert g["gap_s"].iloc[-1] == pytest.approx(last["VAL"] - last["KOR"])


def test_pace_table(model, result):
    t = driver_pace_table(model, result.laps)
    assert len(t) == result.laps["driver"].nunique()
    assert t["model_pace_s"].is_monotonic_increasing


def test_needs_enough_data(result):
    with pytest.raises(ValueError):
        LapDeltaModel(result.circuit).fit(result.laps.head(10))


def test_unraced_compound_uses_prior(result):
    no_soft = result.laps[result.laps["compound"] != "SOFT"]
    m = LapDeltaModel(result.circuit).fit(no_soft)
    assert "SOFT" not in m.fitted_from_data
    assert m.comp_offset["SOFT"] == COMPOUNDS["SOFT"].pace_offset_s
