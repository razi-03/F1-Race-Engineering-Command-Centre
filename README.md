# 🏁 F1 Race Engineering Command Centre

An interactive **Streamlit + Plotly** pit-wall dashboard powered by a Python **ETL pipeline**.
It collects live weather and track data, fits a **lap-delta model** that separates driver pace from
fuel, tyres and track temperature, and turns the result into **actionable strategy calls**.

> Unofficial engineering demo. Not affiliated with Formula 1, the FIA or any team. The grid is fictional.

![python](https://img.shields.io/badge/python-3.10%2B-blue) ![streamlit](https://img.shields.io/badge/streamlit-1.50%2B-red) ![license](https://img.shields.io/badge/license-MIT-green)

---

## What it does

| Tab | What you get |
|---|---|
| 🏁 **Pit Wall** | KPI strip (track temp, rain risk, optimal plan, pit loss, model fit), prioritised strategy calls, top-6 stint plans and the race position chart |
| ⏱️ **Lap Delta** | Cumulative gap between any two drivers, fuel- and temperature-corrected pace traces, a **waterfall that explains a single lap's delta** (driver / compound / tyre age / fuel / track temp / unexplained) and a fuel-corrected pace ranking |
| 🛞 **Tyres** | Fitted degradation curves per compound with observed laps overlaid, cliff markers, crossover laps and stint history |
| 🧠 **Strategy** | Exhaustive 1- and 2-stop optimiser with what-if sliders (pit loss, track temperature), pit windows, race-trace chart and an **undercut calculator** |
| 🌦️ **Weather** | 72-hour forecast for the circuit (air temp, estimated track temp, rain probability) with the race window highlighted |
| ⚙️ **Pipeline** | ETL step log with timings, data-quality report, model coefficients, residuals, SQLite run history and CSV downloads |

## Architecture

```
            ┌─────────────── EXTRACT ───────────────┐
 Open-Meteo │ hourly weather (live, no API key)     │──┐
 API        │   ↳ synthetic fallback when offline    │  │
 config.py  │ circuit catalogue (13 tracks)         │  │
 sim / CSV  │ lap timing: simulator or your CSV     │  │
            └───────────────────────────────────────┘  │
                                                        ▼
            ┌────────────── TRANSFORM ──────────────┐
            │ track-temp estimate (air + solar)     │
            │ per-lap weather join, fuel load       │
            │ positions & gaps, clean-lap flags     │
            │ (pit, SC, outliers), DQ report        │
            └───────────────────────────────────────┘
                                │                │
                                ▼                ▼
            ┌──────── LOAD ────────┐   ┌──────── ANALYTICS ────────┐
            │ SQLite warehouse     │   │ LapDeltaModel (ridge LS)  │
            │ laps / weather /     │   │ StrategyEngine (1-2 stop) │
            │ pipeline_runs        │   │ undercut + rain crossover │
            └──────────────────────┘   │ insight generator         │
                                       └─────────────┬─────────────┘
                                                     ▼
                                        Streamlit + Plotly dashboard
```

### The lap-delta model

Every clean lap is decomposed as

```
lap_time = base + driver_pace + compound_offset
         + deg_lin[c]·age + deg_quad[c]·age²
         + fuel_coef·fuel_kg + temp_coef·(track_temp − 35 °C) + ε
```

It's fitted with ridge-regularised least squares in pure NumPy. Fuel burn and track temperature both
trend steadily through a race, so they are strongly collinear. Those two terms are shrunk towards
physical priors (≈0.032 s/kg, ≈0.018 s/°C), which keeps the decomposition physically meaningful.
A compound nobody raced falls back to circuit-scaled priors and is labelled `prior` in the UI.

### The strategy engine

Fuel, base pace and temperature cost the same whatever the tyre plan. So the race time difference
between strategies reduces to per-stint tyre cost + pit-lane loss + out-lap warm-up. Stint costs are
pre-computed cumulative sums, which makes **every** legal 1-stop and 2-stop plan (two-compound rule,
minimum stint length, compound life) cheap to evaluate with vectorised NumPy.

## Quick start

```bash
git clone https://github.com/<you>/f1-race-engineering-command-centre.git
cd f1-race-engineering-command-centre
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. No API keys are needed.

### Headless pipeline (cron / CI)

```bash
python scripts/run_etl.py --circuit monza --start 15:00
python scripts/run_etl.py --circuit spa --offline --export data/spa_laps.csv
python scripts/run_etl.py --csv data/sample_laps.csv --circuit silverstone
```

### Use your own timing data

Pick **Upload CSV** in the sidebar. Required columns:

| column | example | notes |
|---|---|---|
| `driver` | `VAL` | any code |
| `lap` | `12` | 1-based |
| `lap_time_s` | `91.482` | seconds |
| `compound` | `MEDIUM` | SOFT / MEDIUM / HARD |
| `tyre_age` | `12` | laps on this set |

Optional: `team`, `stint`, `pit_in`, `pit_out`, `track_status` (`GREEN`/`SC`).
See [`data/sample_laps.csv`](data/sample_laps.csv). Data exported from FastF1 maps directly onto
this schema (`LapTime` → seconds, `TyreLife` → `tyre_age`).

## Deploy for free (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. Go to <https://share.streamlit.io> → **New app** → pick the repo, branch `main`, file `app.py`.
3. Deploy. `requirements.txt` and `.streamlit/config.toml` are picked up automatically.

## Development

```bash
pip install -r requirements-dev.txt
pytest -q          # 28 tests: ETL, model, strategy and a headless UI test (Streamlit AppTest)
ruff check .
```

Set `F1CC_OFFLINE=1` to start the dashboard with the synthetic weather source (this is what CI does).
GitHub Actions runs lint, tests and an ETL smoke run on Python 3.11 and 3.12.

## Project layout

```
├── app.py                    # Streamlit dashboard
├── f1cc/
│   ├── config.py             # circuits, compounds, constants
│   ├── etl/
│   │   ├── extract.py        # Open-Meteo, lap simulator, CSV reader
│   │   ├── transform.py      # cleaning, enrichment, joins, DQ report
│   │   ├── load.py           # SQLite warehouse
│   │   └── pipeline.py       # orchestration + step log
│   └── analytics/
│       ├── lap_model.py      # LapDeltaModel, gap & delta decomposition
│       ├── strategy.py       # StrategyEngine, undercut, rain crossover
│       └── insights.py       # pit-wall call generator
├── scripts/run_etl.py        # CLI
├── tests/                    # pytest suite
├── data/sample_laps.csv      # example upload
└── .github/workflows/ci.yml
```

## Limitations and ideas

- The track-temperature estimate is a linear air + solar proxy, not a measured value.
- Traffic, DRS trains and blue flags are left in the *unexplained* term.
- Next steps: a FastF1 loader, safety-car probability in the strategy Monte Carlo, wet-compound modelling.

## License

MIT. See [LICENSE](LICENSE). Weather data © [Open-Meteo](https://open-meteo.com) (CC BY 4.0).
