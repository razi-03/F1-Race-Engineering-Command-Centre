"""F1 Race Engineering Command Centre - Streamlit dashboard.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import os
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from f1cc import __version__
from f1cc.analytics.insights import generate_insights
from f1cc.analytics.lap_model import LapDeltaModel, cumulative_gap, delta_decomposition, driver_pace_table
from f1cc.analytics.strategy import StrategyEngine, rain_assessment, undercut
from f1cc.config import CIRCUITS, COMPOUNDS, TEAM_COLORS
from f1cc.etl.extract import ExtractError
from f1cc.etl.load import recent_runs
from f1cc.etl.pipeline import run_pipeline

st.set_page_config(page_title="F1 Race Engineering Command Centre", page_icon="🏁", layout="wide",
                   initial_sidebar_state="expanded")

ACCENT = "#E10600"
COMP_COLORS = {c: spec.color for c, spec in COMPOUNDS.items()}
LEVEL_STYLE = {"critical": ("#FF3B30", "CRITICAL"), "warning": ("#FF9F0A", "WATCH"),
               "info": ("#0A84FF", "INFO"), "success": ("#30D158", "CALL")}

st.markdown(
    f"""
    <style>
      .block-container {{padding-top: 2.2rem; padding-bottom: 2rem;}}
      .cc-hero {{display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:.6rem;
                border-bottom:2px solid {ACCENT}; padding-bottom:.6rem; margin-bottom:1rem;}}
      .cc-hero h1 {{font-size:1.6rem; margin:0; padding:0; letter-spacing:.03em;}}
      .cc-hero .sub {{color:#9AA0A6; font-size:.9rem;}}
      .cc-pill {{display:inline-block; padding:.15rem .6rem; border-radius:999px; font-size:.75rem;
                font-weight:600; background:#1F2430; color:#E8EAED; margin-left:.35rem;}}
      .cc-live {{background:{ACCENT}; color:white;}}
      .cc-card {{background:#151922; border:1px solid #232938; border-left:4px solid var(--c);
                border-radius:8px; padding:.7rem .9rem; margin-bottom:.55rem;}}
      .cc-card .tag {{font-size:.68rem; font-weight:700; letter-spacing:.08em; color:var(--c);}}
      .cc-card .title {{font-weight:650; font-size:1rem; margin:.1rem 0 .2rem; color:#F2F3F5;}}
      .cc-card .detail {{color:#B8BCC4; font-size:.87rem; line-height:1.4;}}
      div[data-testid="stMetric"] {{background:#151922; border:1px solid #232938; border-radius:8px;
                                    padding:.6rem .8rem;}}
      div[data-testid="stMetricLabel"] p {{font-size:.76rem; color:#9AA0A6; text-transform:uppercase;
                                          letter-spacing:.05em;}}
    </style>
    """,
    unsafe_allow_html=True,
)


def style_fig(fig: go.Figure, height: int = 380, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height, margin=dict(l=10, r=10, t=56 if title else 30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text=title or "", font=dict(size=14)),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0), font=dict(size=12),
    )
    fig.update_xaxes(gridcolor="#232938")
    fig.update_yaxes(gridcolor="#232938")
    return fig


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 🏁 Session setup")
    circuit_key = st.selectbox("Circuit", list(CIRCUITS), index=0,
                               format_func=lambda k: f"{CIRCUITS[k].name} ({CIRCUITS[k].country})")
    c1, c2 = st.columns(2)
    race_day = c1.date_input("Race day", value=date.today(), min_value=date.today(),
                             max_value=date.today() + timedelta(days=2))
    race_time = c2.time_input("Start (local)", value=time(15, 0), step=1800)
    live = st.toggle("Live weather (Open-Meteo)", value=os.environ.get("F1CC_OFFLINE") != "1",
                     help="Falls back to a synthetic forecast automatically if the API is unreachable.")

    st.markdown("### 📥 Lap data source")
    source = st.radio("Source", ["Simulated race", "Upload CSV"], label_visibility="collapsed")
    uploaded, seed = None, 7
    if source == "Simulated race":
        seed = st.number_input("Simulation seed", 1, 9999, 7, help="Change to generate a different race.")
    else:
        up = st.file_uploader("Lap timing CSV", type="csv",
                              help="Columns: driver, lap, lap_time_s, compound, tyre_age "
                                   "(optional: team, stint, pit_in, pit_out, track_status)")
        uploaded = up.getvalue() if up else None
    persist = st.toggle("Persist run to SQLite warehouse", value=True)
    if st.button("🔄 Re-run ETL pipeline", width="stretch"):
        st.cache_data.clear()
    st.caption(f"v{__version__} · unofficial engineering demo · not affiliated with Formula 1")


@st.cache_data(ttl=900, show_spinner=False)
def load_bundle(circuit_key, race_start, live, seed, csv_bytes, persist):
    result = run_pipeline(circuit_key, race_start, live_weather=live, seed=int(seed),
                          laps_csv=csv_bytes, persist=persist)
    model = LapDeltaModel(result.circuit).fit(result.laps)
    return result, model


race_start = datetime.combine(race_day, race_time)
if source == "Upload CSV" and uploaded is None:
    st.info("Upload a lap-timing CSV in the sidebar, or switch back to **Simulated race**.")
    st.stop()

try:
    with st.spinner("Running ETL pipeline and fitting lap model..."):
        R, M = load_bundle(circuit_key, race_start, live, seed, uploaded, persist)
except (ExtractError, ValueError) as exc:
    st.error(f"Pipeline failed: {exc}")
    st.stop()

circuit = R.circuit
laps = R.laps
drivers = sorted(laps["driver"].unique(), key=lambda d: M.driver_pace.get(d, 0))
pace = driver_pace_table(M, laps)
rain = rain_assessment(R.race_weather, race_start, circuit.base_lap_s, circuit.laps)
race_track_temp = float(R.race_weather["track_temp_c"].mean())

# --------------------------------------------------------------------------- header
is_live = "Open-Meteo" in R.weather_note
st.markdown(
    f"""<div class="cc-hero"><div><h1>RACE ENGINEERING COMMAND CENTRE</h1>
    <div class="sub">{circuit.name} · {circuit.country} · {circuit.laps} laps · {circuit.length_km:.3f} km ·
    lights out {race_start:%a %d %b %H:%M} local</div></div>
    <div><span class="cc-pill {'cc-live' if is_live else ''}">{'● LIVE WX' if is_live else 'SIM WX'}</span>
    <span class="cc-pill">{laps.attrs.get('source', 'laps')} laps</span>
    <span class="cc-pill">R² {M.r2:.2f}</span></div></div>""",
    unsafe_allow_html=True,
)
if not is_live and live:
    st.warning(R.weather_note, icon="📡")

tabs = st.tabs(["🏁 Pit Wall", "⏱️ Lap Delta", "🛞 Tyres", "🧠 Strategy", "🌦️ Weather", "⚙️ Pipeline"])

# Strategy what-ifs live on the Strategy tab but feed every tab, so build them first.
with tabs[3]:
    st.markdown("#### What-if controls")
    w1, w2, w3 = st.columns(3)
    pit_loss = w1.slider("Pit-lane loss (s)", 10.0, 35.0, float(circuit.pit_loss_s), 0.5,
                         help="Drop to ~55% of normal to model a stop under safety car.")
    what_if_temp = w2.slider("Track temperature (°C)", 15.0, 60.0, float(round(race_track_temp, 1)), 0.5)
    max_stops = w3.segmented_control("Max stops", [1, 2], default=2) or 2
engine = StrategyEngine(M, circuit, pit_loss_s=pit_loss, track_temp_c=what_if_temp)
strategies = engine.enumerate(max_stops=max_stops)
best = strategies.iloc[0]
insights = generate_insights(M, circuit, strategies, R.race_weather, rain, pace)

# =========================================================================== PIT WALL
with tabs[0]:
    k = st.columns(6)
    nearest = (R.race_weather["time"] - race_start).abs().to_numpy().argmin()
    now_row = R.race_weather.iloc[int(nearest)]
    k[0].metric("Track temp @ start", f"{now_row['track_temp_c']:.0f} °C", f"air {now_row['air_temp_c']:.0f} °C",
                delta_color="off")
    k[1].metric("Rain risk (race)", f"{rain['max_prob']:.0f} %",
                f"crossover ~L{rain['crossover_lap']}" if rain["crossover_lap"] else "dry race",
                delta_color="inverse" if rain["crossover_lap"] else "off")
    k[2].metric("Optimal strategy", best["strategy"].replace(" > ", "-"),
                f"box L{', L'.join(map(str, best['pit_laps']))}", delta_color="off",
                help=" → ".join(f"{c} ({n} laps)" for c, n in zip(best["compounds"], best["stints"])))
    gap2 = strategies.iloc[1]["delta_s"] if len(strategies) > 1 else 0.0
    k[3].metric("Margin to next plan", f"{gap2:.1f} s",
                strategies.iloc[1]["strategy"] if len(strategies) > 1 else "", delta_color="off")
    k[4].metric("Pit loss", f"{pit_loss:.1f} s", f"overtaking {circuit.overtaking:.0%}", delta_color="off")
    k[5].metric("Model fit", f"R² {M.r2:.2f}", f"±{M.rmse:.2f} s/lap", delta_color="off")

    left, right = st.columns([1.05, 1])
    with left:
        st.markdown("#### 📻 Strategy calls")
        for ins in insights:
            color, tag = LEVEL_STYLE[ins.level]
            st.markdown(f"""<div class="cc-card" style="--c:{color}"><div class="tag">{tag}</div>
                        <div class="title">{ins.title}</div><div class="detail">{ins.detail}</div></div>""",
                        unsafe_allow_html=True)
    with right:
        st.markdown("#### 🗺️ Top strategies (stint plan)")
        top = strategies.head(6).iloc[::-1]
        fig = go.Figure()
        for _, row in top.iterrows():
            start_lap = 0
            label = f"#{row['rank']} {row['strategy']} (+{row['delta_s']:.1f}s)"
            for comp, n in zip(row["compounds"], row["stints"]):
                fig.add_trace(go.Bar(y=[label], x=[n], base=start_lap, orientation="h",
                                     marker_color=COMP_COLORS[comp], marker_line=dict(color="#0B0D12", width=2),
                                     showlegend=False,
                                     hovertemplate=f"{comp}: laps {start_lap + 1}-{start_lap + n}<extra></extra>",
                                     text=f"{comp[0]} {n}", textposition="inside", insidetextanchor="middle",
                                     textfont=dict(color="#111")))
                start_lap += n
        fig.update_layout(barmode="stack")
        fig.update_xaxes(title="Lap")
        st.plotly_chart(style_fig(fig, 330), width="stretch")

        st.markdown("#### 📈 Race positions")
        figp = go.Figure()
        for i, d in enumerate(drivers):
            dd = laps[laps["driver"] == d]
            team = dd["team"].iloc[0]
            figp.add_trace(go.Scatter(x=dd["lap"], y=dd["position"], name=d, mode="lines",
                                      line=dict(color=TEAM_COLORS.get(team, "#888"), width=2,
                                                dash="dot" if i % 2 else "solid")))
        figp.update_yaxes(autorange="reversed", dtick=1, title="Pos")
        figp.update_xaxes(title="Lap")
        st.plotly_chart(style_fig(figp, 360), width="stretch")

# =========================================================================== LAP DELTA
with tabs[1]:
    s1, s2, s3 = st.columns([1, 1, 1.3])
    a = s1.selectbox("Driver A", drivers, index=0)
    b = s2.selectbox("Driver B (reference)", drivers, index=min(1, len(drivers) - 1))
    s3.write("")
    corrected = s3.toggle("Fuel & track-temp corrected lap times", value=True)
    if a == b:
        st.info("Pick two different drivers to compare.")
    else:
        g = cumulative_gap(laps, a, b)
        c_left, c_right = st.columns(2)
        with c_left:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=g["lap"], y=g["gap_s"], mode="lines", line=dict(color=ACCENT, width=3),
                                     fill="tozeroy", fillcolor="rgba(225,6,0,0.12)", name=f"{a} vs {b}"))
            for col, who, sym, colr in (("pit_in_a", a, "triangle-down", "#FFD12E"),
                                        ("pit_in_b", b, "triangle-up", "#0A84FF")):
                p = g[g[col]]
                fig.add_trace(go.Scatter(x=p["lap"], y=p["gap_s"], mode="markers", name=f"{who} pits",
                                         marker=dict(symbol=sym, size=11, color=colr)))
            fig.add_hline(y=0, line_color="#666")
            fig.update_yaxes(title=f"Gap (s), + = {a} behind")
            fig.update_xaxes(title="Lap")
            st.plotly_chart(style_fig(fig, 380, f"Cumulative race gap: {a} to {b}"), width="stretch")
        with c_right:
            fig = go.Figure()
            for d, color in ((a, ACCENT), (b, "#0A84FF")):
                dd = laps[(laps["driver"] == d) & laps["is_clean"]]
                y = M.fuel_corrected(dd) if corrected else dd["lap_time_s"]
                fig.add_trace(go.Scatter(
                    x=dd["lap"], y=y, mode="markers+lines", name=d, line=dict(color=color, width=1.5),
                    marker=dict(size=7, color=[COMP_COLORS[c] for c in dd["compound"]],
                                line=dict(color=color, width=1.5))))
            fig.update_yaxes(title="Lap time (s)" + (", corrected" if corrected else ""))
            fig.update_xaxes(title="Lap")
            st.plotly_chart(style_fig(fig, 380, "Clean-lap pace (marker fill = compound)"), width="stretch")

        common = sorted(set(laps.loc[laps["driver"] == a, "lap"]) & set(laps.loc[laps["driver"] == b, "lap"]))
        lap_sel = st.slider("Explain the lap-time delta on lap", int(common[0]), int(common[-1]),
                            int(common[len(common) // 2]))
        dec = delta_decomposition(M, laps, a, b, lap_sel)
        actual = dec.attrs["actual"]
        c_w, c_t = st.columns([1.4, 1])
        with c_w:
            fig = go.Figure(go.Waterfall(
                x=[*dec["component"], "actual delta"], y=[*dec["delta_s"], actual],
                measure=["relative"] * len(dec) + ["total"],
                increasing=dict(marker_color="#FF453A"), decreasing=dict(marker_color="#30D158"),
                totals=dict(marker_color=ACCENT), connector=dict(line=dict(color="#555")),
                text=[f"{v:+.3f}" for v in [*dec["delta_s"], actual]], textposition="outside",
                cliponaxis=False,
            ))
            fig.update_yaxes(title=f"s (+ = {a} slower)")
            verdict = "slower" if actual > 0 else "faster"
            st.plotly_chart(style_fig(fig, 380, f"Lap {lap_sel}: why is {a} {abs(actual):.3f}s {verdict} than {b}?"),
                            width="stretch")
        with c_t:
            ra = laps[(laps["driver"] == a) & (laps["lap"] == lap_sel)].iloc[0]
            rb = laps[(laps["driver"] == b) & (laps["lap"] == lap_sel)].iloc[0]
            st.dataframe(pd.DataFrame({
                a: [f"{ra.lap_time_s:.3f}", ra.compound, ra.tyre_age, f"{ra.fuel_kg:.1f}", ra.position],
                b: [f"{rb.lap_time_s:.3f}", rb.compound, rb.tyre_age, f"{rb.fuel_kg:.1f}", rb.position],
            }, index=["lap time (s)", "compound", "tyre age", "fuel (kg)", "position"]).astype(str), width="stretch")
            st.caption("Green bars help driver A, red bars cost driver A. *Unexplained* covers traffic, "
                       "mistakes, DRS and pit activity the model does not capture.")

    st.markdown("#### 🏎️ Fuel-corrected pace ranking")
    st.dataframe(pace, width="stretch", hide_index=True, column_config={
        "model_pace_s": st.column_config.NumberColumn("Pace vs field (s/lap)", format="%+.3f"),
        "median_fc_lap_s": st.column_config.NumberColumn("Median corrected lap (s)", format="%.3f"),
        "best_lap_s": st.column_config.NumberColumn("Best lap (s)", format="%.3f"),
        "consistency_s": st.column_config.NumberColumn("Consistency σ (s)", format="%.3f"),
    })

# =========================================================================== TYRES
with tabs[2]:
    t1, t2 = st.columns(2)
    with t1:
        fig = go.Figure()
        clean = laps[laps["is_clean"]].copy()
        clean["tyre_effect"] = M.fuel_corrected(clean) - clean["driver"].map(M.driver_pace) - M.base
        for c in COMPOUNDS:
            ages = np.arange(1, int(COMPOUNDS[c].life_laps * 1.5) + 1)
            curve = M.comp_offset[c] + M.tyre_deg(c, ages)
            obs = clean[clean["compound"] == c]
            if not obs.empty:
                fig.add_trace(go.Scatter(x=obs["tyre_age"], y=obs["tyre_effect"], mode="markers",
                                         marker=dict(color=COMP_COLORS[c], size=4, opacity=0.35), showlegend=False,
                                         hoverinfo="skip"))
            fitted = c in M.fitted_from_data
            fig.add_trace(go.Scatter(x=ages, y=curve, mode="lines", name=c + ("" if fitted else " (prior)"),
                                     line=dict(color=COMP_COLORS[c], width=3, dash="solid" if fitted else "dash")))
            fig.add_vline(x=COMPOUNDS[c].life_laps, line=dict(color=COMP_COLORS[c], dash="dot", width=1))
        fig.update_xaxes(title="Tyre age (laps)")
        fig.update_yaxes(title="Time vs new SOFT (s)", range=[-0.5, 7])
        st.plotly_chart(style_fig(fig, 430, "Degradation model (dots = corrected laps, dotted = cliff)"),
                        width="stretch")
    with t2:
        stints = (laps.groupby(["driver", "stint", "compound"], as_index=False)
                  .agg(start=("lap", "min"), end=("lap", "max")))
        fig = go.Figure()
        for _, s in stints.iterrows():
            fig.add_trace(go.Bar(
                y=[s["driver"]], x=[s["end"] - s["start"] + 1], base=s["start"] - 1, orientation="h",
                marker_color=COMP_COLORS[s["compound"]], marker_line=dict(color="#0B0D12", width=2), showlegend=False,
                hovertemplate=f"{s['driver']} {s['compound']}: L{s['start']}-L{s['end']}<extra></extra>"))
        fig.update_layout(barmode="overlay")
        fig.update_yaxes(categoryorder="array", categoryarray=drivers[::-1])
        fig.update_xaxes(title="Lap")
        st.plotly_chart(style_fig(fig, 430, "Race stint history"), width="stretch")

    deg_tbl = pd.DataFrame([{
        "compound": c,
        "source": "fitted" if c in M.fitted_from_data else "prior",
        "pace offset vs SOFT (s)": round(M.comp_offset[c], 3),
        "loss @10 laps (s)": round(float(M.tyre_deg(c, 10)), 3),
        "loss @20 laps (s)": round(float(M.tyre_deg(c, 20)), 3),
        "cliff (lap)": COMPOUNDS[c].life_laps,
        "faster than SOFT from lap": next((x for x in range(1, 60)
                                           if c != "SOFT" and M.comp_offset[c] + M.tyre_deg(c, x)
                                           < M.tyre_deg("SOFT", x)), "–"),
    } for c in COMPOUNDS])
    st.dataframe(deg_tbl, width="stretch", hide_index=True)

# =========================================================================== STRATEGY
with tabs[3]:
    s_left, s_right = st.columns([1.1, 1])
    with s_left:
        st.markdown("#### 🏆 Strategy ranking")
        show = strategies.head(12).copy()
        show["plan"] = [" → ".join(f"{c[0]}{n}" for c, n in zip(cs, ns))
                        for cs, ns in zip(show["compounds"], show["stints"])]
        show["pit laps"] = [", ".join(map(str, p)) for p in show["pit_laps"]]
        show["window"] = [f"L{w[0]}–L{w[1]}" if w else "–" for w in show["pit_window"]]
        st.dataframe(show[["rank", "strategy", "stops", "plan", "pit laps", "window", "delta_s"]], hide_index=True,
                     width="stretch", column_config={"delta_s": st.column_config.ProgressColumn(
                         "Δ to best (s)", format="+%.1f", min_value=0.0,
                         max_value=float(max(show["delta_s"].max(), 1.0)))})
        est = engine.absolute_race_time(list(zip(best["compounds"], best["stints"])))
        st.caption(f"Predicted race time for the optimal plan: **{int(est // 3600)}h {int(est % 3600 // 60):02d}m "
                   f"{est % 60:04.1f}s** at {what_if_temp:.1f} °C track, {pit_loss:.1f}s pit loss.")
    with s_right:
        st.markdown("#### 📉 Race trace vs optimal")
        ref = np.cumsum(engine.lap_trace(best["compounds"], best["pit_laps"]))
        fig = go.Figure()
        palette = [ACCENT, "#0A84FF", "#30D158", "#FFD12E", "#BF5AF2", "#64D2FF"]
        for i, (_, r) in enumerate(strategies.head(6).iterrows()):
            tr = np.cumsum(engine.lap_trace(r["compounds"], r["pit_laps"]))
            fig.add_trace(go.Scatter(x=np.arange(1, len(tr) + 1), y=ref - tr, mode="lines", name=r["strategy"],
                                     line=dict(color=palette[i % len(palette)], width=3 if i == 0 else 2)))
        fig.update_yaxes(title="Gap to optimal plan (s, + = ahead)")
        fig.update_xaxes(title="Lap")
        st.plotly_chart(style_fig(fig, 400), width="stretch")

    st.markdown("#### ⚔️ Undercut calculator")
    u = st.columns(6)
    gap_in = u[0].number_input("Gap to car ahead (s)", 0.0, 10.0, 1.5, 0.1)
    r_comp = u[1].selectbox("Rival compound", list(COMPOUNDS), index=1)
    r_age = u[2].number_input("Rival tyre age", 1, 60, 18)
    my_new = u[3].selectbox("I fit", list(COMPOUNDS), index=2)
    r_new = u[4].selectbox("Rival fits", list(COMPOUNDS), index=2)
    stay = u[5].number_input("Laps rival stays out", 1, 5, 1)
    res = undercut(M, gap_in, r_comp, int(r_age), my_new, r_new, int(stay), deg_factor=engine.deg_factor)
    (st.success if res["success"] else st.error)(
        f"**{res['verdict']}** · fresh-tyre gain {res['gain_s']:+.2f}s over {int(stay) + 1} laps")
    st.dataframe(res["table"], hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f")
                                for c in ("me_s", "rival_s", "gain_s", "cum_gain_s")})

# =========================================================================== WEATHER
with tabs[4]:
    W = R.weather
    fig = go.Figure()
    fig.add_trace(go.Bar(x=W["time"], y=W["rain_prob_pct"], name="Rain probability %", yaxis="y2",
                         marker_color="rgba(10,132,255,0.35)"))
    fig.add_trace(go.Scatter(x=W["time"], y=W["track_temp_c"], name="Track temp °C",
                             line=dict(color=ACCENT, width=3)))
    fig.add_trace(go.Scatter(x=W["time"], y=W["air_temp_c"], name="Air temp °C", line=dict(color="#FFD12E", width=2)))
    race_end = race_start + timedelta(seconds=circuit.laps * circuit.base_lap_s * 1.05)
    fig.add_vrect(x0=race_start, x1=race_end, fillcolor="rgba(255,255,255,0.08)", line_width=0,
                  annotation_text="RACE", annotation_position="top left")
    fig.update_layout(yaxis=dict(title="°C"),
                      yaxis2=dict(title="Rain %", overlaying="y", side="right", range=[0, 100], showgrid=False))
    st.plotly_chart(style_fig(fig, 430, R.weather_note), width="stretch")

    wc = st.columns(4)
    rw = R.race_weather
    wc[0].metric("Track temp (race avg)", f"{rw['track_temp_c'].mean():.1f} °C",
                 f"{rw['track_temp_c'].iloc[-1] - rw['track_temp_c'].iloc[0]:+.1f} °C trend")
    wc[1].metric("Humidity", f"{rw['humidity_pct'].mean():.0f} %")
    wc[2].metric("Wind", f"{rw['wind_kph'].mean():.0f} km/h", f"from {rw['wind_dir_deg'].mean():.0f}°",
                 delta_color="off")
    wc[3].metric("Max rain", f"{rw['precip_mm'].max():.1f} mm/h", rw["condition"].mode().iloc[0], delta_color="off")
    st.dataframe(rw, hide_index=True, width="stretch",
                 column_config={"time": st.column_config.DatetimeColumn(format="ddd HH:mm")})

# =========================================================================== PIPELINE
with tabs[5]:
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("#### 🔧 ETL run log")
        st.dataframe(pd.DataFrame(R.log), hide_index=True, width="stretch")
        st.markdown("#### ✅ Data quality")
        q = R.report
        qc = st.columns(4)
        qc[0].metric("Valid laps", q["rows_valid"], f"-{q['rows_dropped']} dropped", delta_color="off")
        qc[1].metric("Clean laps", q["clean_laps"])
        qc[2].metric("Pit/SC excluded", q["excluded_pit"] + q["excluded_sc"])
        qc[3].metric("Outliers", q["excluded_outlier"])
        if R.run_id:
            st.caption(f"Persisted as run `{R.run_id}` → `data/warehouse.db`")
            st.dataframe(recent_runs(), hide_index=True, width="stretch")
    with p2:
        st.markdown("#### 🧮 Model coefficients")
        st.dataframe(M.coefficients(), hide_index=True, width="stretch",
                     column_config={"value": st.column_config.NumberColumn(format="%.4f")})
        clean = laps[laps["is_clean"]]
        resid = clean["lap_time_s"] - M.predict(clean)
        fig = go.Figure(go.Histogram(x=resid, nbinsx=40, marker_color=ACCENT))
        fig.update_xaxes(title="Residual (s)")
        st.plotly_chart(style_fig(fig, 280, f"Residuals · n={M.n_obs} · RMSE {M.rmse:.3f}s"), width="stretch")
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Curated laps (CSV)", laps.to_csv(index=False).encode(), f"{circuit.key}_laps.csv",
                       "text/csv", width="stretch")
    d2.download_button("⬇️ Weather forecast (CSV)", R.weather.to_csv(index=False).encode(),
                       f"{circuit.key}_weather.csv", "text/csv", width="stretch")
