import pandas as pd
import pytest

from f1cc.analytics.insights import generate_insights
from f1cc.analytics.lap_model import driver_pace_table
from f1cc.analytics.strategy import StrategyEngine, rain_assessment, undercut


@pytest.fixture(scope="module")
def engine(model, result):
    return StrategyEngine(model, result.circuit)


def test_enumeration_is_sorted_and_legal(engine):
    s = engine.enumerate()
    assert s["race_time_rel_s"].is_monotonic_increasing
    assert s.iloc[0]["delta_s"] == 0
    for comps, stints in zip(s["compounds"], s["stints"]):
        assert len(set(comps)) >= 2                 # two dry compounds rule
        assert sum(stints) == engine.L
        assert min(stints) >= engine.min_stint


def test_enumeration_matches_brute_force(engine):
    s = engine.enumerate(max_stops=1)
    best = s.iloc[0]
    brute = min(
        engine.plan_time([(c1, p), (c2, engine.L - p)])
        for c1 in ("SOFT", "MEDIUM", "HARD") for c2 in ("SOFT", "MEDIUM", "HARD") if c1 != c2
        for p in range(engine.min_stint, engine.L - engine.min_stint + 1)
        if p <= engine.max_stint(c1) and engine.L - p <= engine.max_stint(c2)
    )
    assert best["race_time_rel_s"] == pytest.approx(brute)


def test_higher_pit_loss_favours_fewer_stops(model, result):
    cheap = StrategyEngine(model, result.circuit, pit_loss_s=8).enumerate().iloc[0]
    dear = StrategyEngine(model, result.circuit, pit_loss_s=40).enumerate().iloc[0]
    assert dear["stops"] <= cheap["stops"]


def test_hotter_track_increases_deg(model, result):
    cool = StrategyEngine(model, result.circuit, track_temp_c=20)
    hot = StrategyEngine(model, result.circuit, track_temp_c=55)
    assert hot.deg_factor > cool.deg_factor


def test_lap_trace_sums_to_plan_time(engine):
    best = engine.enumerate().iloc[0]
    assert engine.lap_trace(best["compounds"], best["pit_laps"]).sum() == pytest.approx(best["race_time_rel_s"])


def test_undercut_monotonic_in_gap(model):
    small = undercut(model, 0.3, "MEDIUM", 25, "HARD", "HARD", 2)
    big = undercut(model, 6.0, "MEDIUM", 25, "HARD", "HARD", 2)
    assert small["gain_s"] == pytest.approx(big["gain_s"])
    assert small["final_gap_s"] < big["final_gap_s"]
    assert not big["success"]


def test_rain_assessment(race_start):
    w = pd.DataFrame({"time": pd.date_range(race_start, periods=3, freq="h"),
                      "rain_prob_pct": [10, 70, 90], "precip_mm": [0, 1.2, 3.0]})
    r = rain_assessment(w, race_start, 90.0, 52)
    assert r["max_prob"] == 90
    assert r["crossover_lap"] == 41  # 3600 s / 90 s + 1


def test_insights_contain_a_call(model, result, engine):
    s = engine.enumerate()
    rain = rain_assessment(result.race_weather, result.race_start, result.circuit.base_lap_s, result.circuit.laps)
    ins = generate_insights(model, result.circuit, s, result.race_weather, rain, driver_pace_table(model, result.laps))
    assert any(i.level == "success" and "Optimal plan" in i.title for i in ins)
