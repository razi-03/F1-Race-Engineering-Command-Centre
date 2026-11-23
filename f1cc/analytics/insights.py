"""Turns model + strategy output into short, actionable pit-wall calls."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from f1cc.analytics.lap_model import LapDeltaModel
from f1cc.config import COMPOUNDS, Circuit


@dataclass
class Insight:
    level: str      # "critical" | "warning" | "info" | "success"
    title: str
    detail: str


def generate_insights(model: LapDeltaModel, circuit: Circuit, strategies: pd.DataFrame,
                      race_weather: pd.DataFrame, rain: dict, pace: pd.DataFrame) -> list[Insight]:
    out: list[Insight] = []

    # 1. Rain risk trumps everything
    if rain.get("crossover_lap"):
        out.append(Insight("critical", f"Rain expected around lap {rain['crossover_lap']}",
                           f"Forecast shows measurable rain at {rain['first_wet_time']:%H:%M}. Prepare intermediates; "
                           "keep the car within one pit-stop window of the crossover and avoid stopping for slicks just before it."))
    elif rain.get("max_prob", 0) >= 40:
        out.append(Insight("warning", f"Rain risk {rain['max_prob']:.0f}% during the race",
                           "Keep strategy flexible: favour a plan whose pit window can move +/-5 laps without big losses."))

    # 2. Optimal strategy
    best = strategies.iloc[0]
    second = strategies.iloc[1] if len(strategies) > 1 else None
    plan = " -> ".join(f"{c} ({n})" for c, n in zip(best["compounds"], best["stints"]))
    margin = f"{second['delta_s']:.1f}s faster than {second['strategy']}" if second is not None else ""
    out.append(Insight("success", f"Optimal plan: {best['stops']}-stop {best['strategy']}",
                       f"{plan}. Box on lap(s) {', '.join(map(str, best['pit_laps']))}. {margin}."))

    one_stop = strategies[strategies["stops"] == 1]
    two_stop = strategies[strategies["stops"] == 2]
    if not one_stop.empty and not two_stop.empty:
        d = two_stop["race_time_rel_s"].min() - one_stop["race_time_rel_s"].min()
        if abs(d) < 3:
            out.append(Insight("info", "1-stop vs 2-stop is marginal",
                               f"Only {abs(d):.1f}s separates them. Track position decides it: "
                               + ("with hard overtaking, protect position with the 1-stop."
                                  if circuit.overtaking < 0.5 else "overtaking is feasible, so the 2-stop is a valid attack option.")))
        if not one_stop.empty and one_stop.iloc[0]["pit_window"]:
            lo, hi = one_stop.iloc[0]["pit_window"]
            out.append(Insight("info", f"1-stop pit window: laps {lo}-{hi}",
                               "Any stop inside this window costs < 1.0s vs the theoretical optimum. Use it to react to "
                               "safety cars and rivals' undercut attempts."))

    # 3. Degradation
    deg_at_15 = {c: float(model.tyre_deg(c, 15)) for c in COMPOUNDS}
    worst = max(deg_at_15, key=deg_at_15.get)
    if deg_at_15[worst] > 1.6:
        out.append(Insight("warning", f"{worst} degradation is high",
                           f"{deg_at_15[worst]:.2f}s lost after 15 laps. Keep {worst} stints short and manage the out-lap."))

    # 4. Track temperature
    if not race_weather.empty:
        t0, t1 = race_weather["track_temp_c"].iloc[0], race_weather["track_temp_c"].iloc[-1]
        if t1 - t0 >= 4:
            out.append(Insight("warning", f"Track warming {t0:.0f} -> {t1:.0f} C",
                               "Thermal degradation rises late in the race; bias towards harder compounds for the final stint."))
        elif t0 - t1 >= 4:
            out.append(Insight("info", f"Track cooling {t0:.0f} -> {t1:.0f} C",
                               "Softer compounds will last longer in the closing stint; consider SOFT for the run to the flag."))

    # 5. Pace
    if not pace.empty and len(pace) > 1:
        fastest = pace.iloc[0]
        out.append(Insight("info", f"Fastest fuel-corrected pace: {fastest['driver']} ({fastest['team']})",
                           f"{abs(pace.iloc[1]['model_pace_s'] - fastest['model_pace_s']):.3f}s/lap clear of {pace.iloc[1]['driver']} "
                           "once fuel, tyres and track temperature are stripped out."))

    if model.r2 < 0.6:
        out.append(Insight("warning", "Low model confidence",
                           f"R2 = {model.r2:.2f}. Lap data is noisy; treat strategy deltas below ~2s as a coin flip."))
    return out
