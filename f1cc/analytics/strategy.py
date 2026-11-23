"""Race-strategy engine built on top of the fitted lap model.

Because fuel, base pace and track temperature cost the same whatever the tyre plan,
race time differences between strategies reduce to:

    sum over stints [ n * compound_offset + sum_{age=1..n} deg(age) ]
    + stops * pit_loss + sum of out-lap warm-up penalties

That makes exhaustive 1- and 2-stop enumeration instant (vectorised NumPy).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from f1cc.analytics.lap_model import LapDeltaModel
from f1cc.config import COMPOUNDS, FUEL_START_KG, TRACK_TEMP_REF_C, Circuit


@dataclass
class StrategyEngine:
    model: LapDeltaModel
    circuit: Circuit
    race_laps: int | None = None
    pit_loss_s: float | None = None
    track_temp_c: float | None = None
    min_stint: int = 5

    def __post_init__(self):
        self.L = self.race_laps or self.circuit.laps
        self.pit_loss = self.pit_loss_s if self.pit_loss_s is not None else self.circuit.pit_loss_s
        self.temp = self.track_temp_c if self.track_temp_c is not None else self.model.mean_track_temp
        # Hotter asphalt -> more thermal degradation (relative to the conditions the model saw).
        self.deg_factor = max(0.5, 1 + 0.012 * (self.temp - self.model.mean_track_temp))
        self._cost = {c: self._stint_cost_table(c) for c in COMPOUNDS}

    # --------------------------------------------------------------- building blocks
    def lap_cost(self, compound: str, ages: np.ndarray) -> np.ndarray:
        return self.model.comp_offset[compound] + self.deg_factor * self.model.tyre_deg(compound, ages)

    def _stint_cost_table(self, compound: str) -> np.ndarray:
        ages = np.arange(1, self.L + 1)
        return np.concatenate([[0.0], np.cumsum(self.lap_cost(compound, ages))])

    def max_stint(self, compound: str) -> int:
        return int(COMPOUNDS[compound].life_laps * 1.6)

    def plan_time(self, plan: list[tuple[str, int]]) -> float:
        """Relative race time of a plan [(compound, laps), ...]."""
        if sum(n for _, n in plan) != self.L:
            raise ValueError("Stints must add up to race distance")
        t = sum(self._cost[c][n] for c, n in plan)
        t += (len(plan) - 1) * self.pit_loss
        t += sum(COMPOUNDS[c].warmup_s for c, _ in plan[1:])
        return float(t)

    def absolute_race_time(self, plan: list[tuple[str, int]]) -> float:
        m = self.model
        fuel = FUEL_START_KG - FUEL_START_KG / self.L * np.arange(self.L)
        const = self.L * m.base + m.fuel_coef * fuel.sum() + self.L * m.temp_coef * (self.temp - TRACK_TEMP_REF_C)
        return const + self.plan_time(plan)

    # ------------------------------------------------------------------ enumeration
    def enumerate(self, max_stops: int = 2) -> pd.DataFrame:
        rows = []
        L, lo = self.L, self.min_stint
        comps = list(COMPOUNDS)
        if max_stops >= 1:
            for c1, c2 in product(comps, repeat=2):
                if c1 == c2:
                    continue
                p = np.arange(lo, L - lo + 1)
                valid = (p <= self.max_stint(c1)) & ((L - p) <= self.max_stint(c2))
                if not valid.any():
                    continue
                t = self._cost[c1][p] + self._cost[c2][L - p] + self.pit_loss + COMPOUNDS[c2].warmup_s
                t = np.where(valid, t, np.inf)
                k = int(np.argmin(t))
                window = p[(t - t[k]) <= 1.0]
                rows.append(self._row([c1, c2], [int(p[k])], float(t[k]), window))
        if max_stops >= 2:
            p1 = np.arange(lo, L - 2 * lo + 1)[:, None]
            p2 = np.arange(2 * lo, L - lo + 1)[None, :]
            for c1, c2, c3 in product(comps, repeat=3):
                if len({c1, c2, c3}) < 2:
                    continue
                n1, n2, n3 = p1, p2 - p1, L - p2
                valid = (n2 >= lo) & (n1 <= self.max_stint(c1)) & (n2 <= self.max_stint(c2)) & (n3 <= self.max_stint(c3))
                if not valid.any():
                    continue
                t = (self._cost[c1][n1] + self._cost[c2][np.clip(n2, 0, L)] + self._cost[c3][n3]
                     + 2 * self.pit_loss + COMPOUNDS[c2].warmup_s + COMPOUNDS[c3].warmup_s)
                t = np.where(valid, t, np.inf)
                i, j = np.unravel_index(int(np.argmin(t)), t.shape)
                rows.append(self._row([c1, c2, c3], [int(p1[i, 0]), int(p2[0, j])], float(t[i, j]), None))
        df = pd.DataFrame(rows).sort_values("race_time_rel_s").reset_index(drop=True)
        df["delta_s"] = df["race_time_rel_s"] - df["race_time_rel_s"].iloc[0]
        df.insert(0, "rank", np.arange(1, len(df) + 1))
        return df

    def _row(self, comps, pits, t, window):
        bounds = [0, *pits, self.L]
        stints = [bounds[i + 1] - bounds[i] for i in range(len(comps))]
        return {
            "strategy": " > ".join(c[0] for c in comps),
            "compounds": comps,
            "stops": len(pits),
            "pit_laps": pits,
            "stints": stints,
            "race_time_rel_s": t,
            "pit_window": (int(window.min()), int(window.max())) if window is not None and len(window) else None,
        }

    # ---------------------------------------------------------------- race traces
    def lap_trace(self, compounds: list[str], pit_laps: list[int]) -> np.ndarray:
        """Per-lap relative lap time for a plan (pit loss booked on the in-lap)."""
        bounds = [0, *pit_laps, self.L]
        out = []
        for s, c in enumerate(compounds):
            n = bounds[s + 1] - bounds[s]
            laps = self.lap_cost(c, np.arange(1, n + 1))
            if s > 0:
                laps[0] += COMPOUNDS[c].warmup_s
            if s < len(compounds) - 1:
                laps[-1] += self.pit_loss
            out.append(laps)
        return np.concatenate(out)


# ---------------------------------------------------------------------- undercut
def undercut(model: LapDeltaModel, gap_s: float, rival_compound: str, rival_age: int,
             my_new: str, rival_new: str, laps_rival_stays_out: int = 1, deg_factor: float = 1.0) -> dict:
    """Project the gap after an undercut attempt.

    I pit now; the rival stays out ``laps_rival_stays_out`` laps on worn tyres, then pits.
    Pit-lane losses cancel out, so the fight is fresh-tyre pace vs worn-tyre pace.
    ``gap_s`` is how far I am behind (positive).
    """
    rows = []
    k = laps_rival_stays_out
    for j in range(1, k + 2):
        me = model.comp_offset[my_new] + deg_factor * float(model.tyre_deg(my_new, j))
        if j == 1:
            me += COMPOUNDS[my_new].warmup_s
        if j <= k:
            rival = model.comp_offset[rival_compound] + deg_factor * float(model.tyre_deg(rival_compound, rival_age + j))
            note = f"rival on {rival_age + j}-lap-old {rival_compound}"
        else:
            rival = (model.comp_offset[rival_new] + deg_factor * float(model.tyre_deg(rival_new, 1))
                     + COMPOUNDS[rival_new].warmup_s)
            note = f"rival out-lap on cold {rival_new}"
        rows.append({"lap": j, "me_s": me, "rival_s": rival, "gain_s": rival - me, "note": note})
    table = pd.DataFrame(rows)
    table["cum_gain_s"] = table["gain_s"].cumsum()
    gain = float(table["gain_s"].sum())
    final_gap = gap_s - gain
    return {
        "table": table,
        "gain_s": gain,
        "final_gap_s": final_gap,
        "success": final_gap < 0,
        "verdict": (f"UNDERCUT WORKS - emerge {abs(final_gap):.2f}s ahead" if final_gap < 0
                    else f"Undercut falls short by {final_gap:.2f}s - consider overcut / extend"),
    }


# ---------------------------------------------------------------------- weather
def rain_assessment(race_weather: pd.DataFrame, race_start, lap_s: float, laps: int) -> dict:
    """Summarise rain risk during the race and estimate the lap a crossover may be needed."""
    if race_weather.empty:
        return {"max_prob": 0.0, "crossover_lap": None, "first_wet_time": None}
    max_prob = float(race_weather["rain_prob_pct"].max())
    wet = race_weather[(race_weather["precip_mm"] >= 0.3) & (race_weather["time"] >= race_start)]
    crossover_lap, first = None, None
    if not wet.empty:
        first = wet["time"].iloc[0]
        crossover_lap = int(max(1, min(laps, (first - race_start).total_seconds() // lap_s + 1)))
    return {"max_prob": max_prob, "crossover_lap": crossover_lap, "first_wet_time": first}
