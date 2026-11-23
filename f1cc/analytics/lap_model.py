"""Lap-time model and lap-delta analytics.

The model decomposes every clean lap into physically meaningful parts:

    lap_time = base + driver_pace + compound_offset
             + deg_lin[c] * age + deg_quad[c] * age^2
             + fuel_coef * fuel_kg + temp_coef * (track_temp - ref)
             + noise

It is fitted with ridge-regularised least squares (pure NumPy), which keeps it
fast, explainable and dependency-light. The fitted coefficients drive the
lap-delta decomposition, degradation curves and the strategy engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from f1cc.config import (
    CLIFF_PENALTY_S_PER_LAP2,
    COMPOUNDS,
    FUEL_EFFECT_S_PER_KG,
    TRACK_TEMP_EFFECT_S_PER_C,
    TRACK_TEMP_REF_C,
    Circuit,
)

COMPONENTS = ["driver", "compound", "tyre", "fuel", "track_temp"]


@dataclass
class LapDeltaModel:
    circuit: Circuit
    ridge: float = 1e-4
    prior_strength: float = 0.5
    base: float = 0.0
    fuel_coef: float = FUEL_EFFECT_S_PER_KG
    temp_coef: float = TRACK_TEMP_EFFECT_S_PER_C
    comp_offset: dict[str, float] = field(default_factory=dict)
    deg_lin: dict[str, float] = field(default_factory=dict)
    deg_quad: dict[str, float] = field(default_factory=dict)
    max_age_seen: dict[str, int] = field(default_factory=dict)
    driver_pace: dict[str, float] = field(default_factory=dict)
    fitted_from_data: set[str] = field(default_factory=set)
    r2: float = float("nan")
    rmse: float = float("nan")
    n_obs: int = 0
    mean_track_temp: float = TRACK_TEMP_REF_C

    # ------------------------------------------------------------------ fitting
    @staticmethod
    def _design(df: pd.DataFrame, drivers: list[str], comps: list[str]) -> np.ndarray:
        cols = [(df["driver"] == d).to_numpy(float) for d in drivers]   # driver intercepts
        cols.append(df["fuel_kg"].to_numpy(float))
        cols.append((df["track_temp_c"] - TRACK_TEMP_REF_C).to_numpy(float))
        for c in comps[1:]:                                             # offsets vs reference compound
            cols.append((df["compound"] == c).to_numpy(float))
        age = df["tyre_age"].to_numpy(float)
        for c in comps:                                                 # compound-specific degradation
            m = (df["compound"] == c).to_numpy(float)
            cols.append(m * age)
            cols.append(m * age**2)
        return np.column_stack(cols)

    def fit(self, laps: pd.DataFrame) -> LapDeltaModel:
        df = laps[laps["is_clean"]] if "is_clean" in laps else laps
        if len(df) < 30:
            raise ValueError("Need at least 30 clean laps to fit the lap model")
        drivers = sorted(df["driver"].unique())
        comps = [c for c in COMPOUNDS if c in set(df["compound"])]
        X = self._design(df, drivers, comps)
        y = df["lap_time_s"].to_numpy(float)

        # Fuel burn and track temperature both trend monotonically through a race, so they are
        # strongly collinear. Shrink those two terms towards physical priors (a Bayesian-style
        # ridge) and lightly regularise everything else. Driver intercepts are never penalised.
        nd = len(drivers)
        scale = X.std(axis=0)
        scale[scale == 0] = 1.0
        Xs = X / scale
        prior = np.zeros(X.shape[1])
        prior[nd], prior[nd + 1] = FUEL_EFFECT_S_PER_KG, TRACK_TEMP_EFFECT_S_PER_C
        penalty = np.full(X.shape[1], self.ridge * len(y))
        penalty[:nd] = 0.0
        penalty[nd:nd + 2] = self.prior_strength * len(y)
        prior_s = prior * scale
        beta = np.linalg.solve(Xs.T @ Xs + np.diag(penalty), Xs.T @ y + penalty * prior_s) / scale

        i = 0
        intercepts = dict(zip(drivers, beta[i:i + len(drivers)]))
        i += len(drivers)
        self.fuel_coef = float(beta[i])
        self.temp_coef = float(beta[i + 1])
        i += 2
        offsets = {comps[0]: 0.0}
        for c in comps[1:]:
            offsets[c] = float(beta[i])
            i += 1
        for c in comps:
            self.deg_lin[c] = float(beta[i])
            self.deg_quad[c] = float(beta[i + 1])
            self.max_age_seen[c] = int(df.loc[df["compound"] == c, "tyre_age"].max())
            i += 2
        self.fitted_from_data = set(comps)

        # Express offsets relative to SOFT; use circuit-scaled priors for compounds nobody raced.
        ref_prior = COMPOUNDS[comps[0]].pace_offset_s
        for c, spec in COMPOUNDS.items():
            if c in offsets:
                self.comp_offset[c] = offsets[c] + ref_prior
            else:
                self.comp_offset[c] = spec.pace_offset_s
                self.deg_lin[c] = spec.deg_lin * self.circuit.abrasiveness
                self.deg_quad[c] = spec.deg_quad * self.circuit.abrasiveness
                self.max_age_seen[c] = spec.life_laps
        mean_int = float(np.mean(list(intercepts.values())))
        self.base = mean_int - ref_prior
        self.driver_pace = {d: float(v - mean_int) for d, v in intercepts.items()}

        pred = X @ beta
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        self.r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        self.rmse = float(np.sqrt(ss_res / len(y)))
        self.n_obs = len(y)
        self.mean_track_temp = float(df["track_temp_c"].mean())
        return self

    # ---------------------------------------------------------------- prediction
    def tyre_deg(self, compound: str, age) -> np.ndarray:
        """Time lost to tyre wear at a given stint age (vs a brand-new tyre)."""
        age = np.asarray(age, dtype=float)
        deg = self.deg_lin[compound] * age + self.deg_quad[compound] * age**2
        life = COMPOUNDS[compound].life_laps
        cliff = CLIFF_PENALTY_S_PER_LAP2 * 0.1 * np.clip(age - life, 0, None) ** 2
        return np.maximum(deg, 0.0) + cliff

    def components(self, driver: str | None, compound: str, age: int, fuel_kg: float,
                   track_temp: float) -> dict[str, float]:
        return {
            "driver": self.driver_pace.get(driver, 0.0) if driver else 0.0,
            "compound": self.comp_offset[compound],
            "tyre": float(self.tyre_deg(compound, age)),
            "fuel": self.fuel_coef * fuel_kg,
            "track_temp": self.temp_coef * (track_temp - TRACK_TEMP_REF_C),
        }

    def predict_row(self, driver, compound, age, fuel_kg, track_temp) -> float:
        return self.base + sum(self.components(driver, compound, age, fuel_kg, track_temp).values())

    def predict(self, laps: pd.DataFrame) -> np.ndarray:
        return np.array([
            self.predict_row(r.driver, r.compound, r.tyre_age, r.fuel_kg, r.track_temp_c)
            for r in laps.itertuples(index=False)
        ])

    def fuel_corrected(self, laps: pd.DataFrame) -> pd.Series:
        """Lap time with fuel load and track temperature removed (zero fuel, reference temp)."""
        return laps["lap_time_s"] - self.fuel_coef * laps["fuel_kg"] - self.temp_coef * (
            laps["track_temp_c"] - TRACK_TEMP_REF_C)

    def coefficients(self) -> pd.DataFrame:
        rows = [{"term": "fuel effect (s/kg)", "value": self.fuel_coef, "source": "fitted"},
                {"term": "track temp effect (s/C)", "value": self.temp_coef, "source": "fitted"}]
        for c in COMPOUNDS:
            src = "fitted" if c in self.fitted_from_data else "prior"
            rows += [
                {"term": f"{c} pace offset vs SOFT (s)", "value": self.comp_offset[c], "source": src},
                {"term": f"{c} deg linear (s/lap)", "value": self.deg_lin[c], "source": src},
                {"term": f"{c} deg quadratic (s/lap^2)", "value": self.deg_quad[c], "source": src},
            ]
        return pd.DataFrame(rows)


# --------------------------------------------------------------------- lap deltas
def cumulative_gap(laps: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Race gap of driver A to driver B per lap (positive = A behind)."""
    cols = ["race_time_s", "lap_time_s", "compound", "pit_in"]
    pa = laps[laps["driver"] == a].set_index("lap")[cols]
    pb = laps[laps["driver"] == b].set_index("lap")[cols]
    j = pa.join(pb, lsuffix="_a", rsuffix="_b", how="inner")
    j["gap_s"] = j["race_time_s_a"] - j["race_time_s_b"]
    j["lap_delta_s"] = j["lap_time_s_a"] - j["lap_time_s_b"]
    return j.reset_index()


def delta_decomposition(model: LapDeltaModel, laps: pd.DataFrame, a: str, b: str, lap: int) -> pd.DataFrame:
    """Explain the lap-time delta between two drivers on a given lap."""
    ra = laps[(laps["driver"] == a) & (laps["lap"] == lap)]
    rb = laps[(laps["driver"] == b) & (laps["lap"] == lap)]
    if ra.empty or rb.empty:
        raise ValueError(f"Lap {lap} missing for {a} or {b}")
    ra, rb = ra.iloc[0], rb.iloc[0]
    ca = model.components(a, ra.compound, ra.tyre_age, ra.fuel_kg, ra.track_temp_c)
    cb = model.components(b, rb.compound, rb.tyre_age, rb.fuel_kg, rb.track_temp_c)
    rows = [{"component": k, "delta_s": ca[k] - cb[k]} for k in COMPONENTS]
    actual = float(ra.lap_time_s - rb.lap_time_s)
    explained = sum(r["delta_s"] for r in rows)
    rows.append({"component": "unexplained", "delta_s": actual - explained})
    out = pd.DataFrame(rows)
    out.attrs["actual"] = actual
    return out


def driver_pace_table(model: LapDeltaModel, laps: pd.DataFrame) -> pd.DataFrame:
    clean = laps[laps["is_clean"]].copy()
    clean["fc"] = model.fuel_corrected(clean)
    clean["resid"] = clean["lap_time_s"] - model.predict(clean)
    agg = clean.groupby(["driver", "team"], as_index=False).agg(
        clean_laps=("lap", "count"),
        median_fc_lap_s=("fc", "median"),
        best_lap_s=("lap_time_s", "min"),
        consistency_s=("resid", "std"),
    )
    agg["model_pace_s"] = agg["driver"].map(model.driver_pace)
    return agg.sort_values("model_pace_s").reset_index(drop=True)
