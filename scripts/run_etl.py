"""Headless ETL + strategy run, for cron jobs, CI or quick checks.

Examples:
    python scripts/run_etl.py --circuit monza
    python scripts/run_etl.py --circuit spa --start 14:00 --offline --export data/spa_laps.csv
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from f1cc.analytics.insights import generate_insights  # noqa: E402
from f1cc.analytics.lap_model import LapDeltaModel, driver_pace_table  # noqa: E402
from f1cc.analytics.strategy import StrategyEngine, rain_assessment  # noqa: E402
from f1cc.config import CIRCUITS  # noqa: E402
from f1cc.etl.pipeline import run_pipeline  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--circuit", default="bahrain", choices=sorted(CIRCUITS))
    ap.add_argument("--start", default="15:00", help="local race start HH:MM (today)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--csv", type=Path, help="lap timing CSV instead of a simulated race")
    ap.add_argument("--offline", action="store_true", help="skip the live weather API")
    ap.add_argument("--no-persist", action="store_true", help="do not write to the SQLite warehouse")
    ap.add_argument("--export", type=Path, help="write curated laps to this CSV")
    args = ap.parse_args()

    hh, mm = map(int, args.start.split(":"))
    start = datetime.combine(date.today(), datetime.min.time()).replace(hour=hh, minute=mm)
    csv_bytes = args.csv.read_bytes() if args.csv else None

    r = run_pipeline(args.circuit, start, live_weather=not args.offline, seed=args.seed,
                     laps_csv=csv_bytes, persist=not args.no_persist)
    for entry in r.log:
        print(f"  [{entry['ms']:>7.1f} ms] {entry['step']:<28} {entry['detail']}")
    print(f"  weather: {r.weather_note}")

    model = LapDeltaModel(r.circuit).fit(r.laps)
    strategies = StrategyEngine(model, r.circuit).enumerate()
    rain = rain_assessment(r.race_weather, start, r.circuit.base_lap_s, r.circuit.laps)
    print(f"\nModel R2={model.r2:.3f} RMSE={model.rmse:.3f}s on {model.n_obs} clean laps")
    print("\nTop strategies:")
    print(strategies.head(5)[["rank", "strategy", "stops", "pit_laps", "stints", "delta_s"]].to_string(index=False))
    print("\nPit-wall calls:")
    for ins in generate_insights(model, r.circuit, strategies, r.race_weather, rain, driver_pace_table(model, r.laps)):
        print(f"  [{ins.level.upper():8}] {ins.title}: {ins.detail}")

    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        r.laps.to_csv(args.export, index=False)
        print(f"\nCurated laps written to {args.export}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
