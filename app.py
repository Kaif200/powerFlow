import io
import json
import time
import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from aiopf_core import run_experiment, GRID_NAME

st.set_page_config(page_title="AI-OPF · Intelligent Power Grid Optimization",
                    page_icon="⚡", layout="wide")

# ---------------------------------------------------------------------------
# STYLE
# ---------------------------------------------------------------------------
st.markdown("""
<style>
:root {
  --bg: #080D14; --panel: #0D1420; --line: #1E293B;
  --cyan: #22D3EE; --amber: #F59E0B; --red: #F87171; --green: #34D399;
}
.stApp { background-color: var(--bg); }
.aiopf-hero { padding: 1.2rem 0 0.4rem 0; }
.aiopf-eyebrow { font-family: monospace; letter-spacing: .25em; color: var(--cyan);
  font-size: 11px; text-transform: uppercase; margin-bottom: 4px;}
.aiopf-title { font-size: 2.6rem; font-weight: 700; color: #F1F5F9; margin: 0;}
.aiopf-sub { color: #94A3B8; max-width: 640px; }
.aiopf-badge { font-family: monospace; font-size: 11px; padding: 3px 10px; border-radius: 999px;
  border: 1px solid var(--line); color: #94A3B8; margin-right: 6px; display:inline-block; }
.metric-card { border: 1px solid var(--line); background: var(--panel); border-radius: 8px;
  padding: 14px 16px; }
.metric-label { font-family: monospace; font-size: 10px; letter-spacing: .2em; text-transform: uppercase;
  color: #64748B; }
.metric-value { font-family: monospace; font-size: 1.6rem; font-weight: 600; color: #E2E8F0; }
.scorecard-row { display:flex; justify-content: space-between; font-family: monospace; font-size: 13px;
  padding: 7px 4px; border-bottom: 1px solid #111827; color: #CBD5E1; }
footer, #MainMenu {visibility: hidden;}
.aiopf-footer { text-align:center; color:#475569; font-size: 12px; padding: 2rem 0 1rem 0; }
.aiopf-footer .name { color: var(--cyan); font-weight: 600; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# HERO
# ---------------------------------------------------------------------------
c1, c2 = st.columns([2, 1])
with c1:
    st.markdown('<div class="aiopf-hero">', unsafe_allow_html=True)
    st.markdown('<div class="aiopf-eyebrow">Intelligent Optimal Power Flow</div>', unsafe_allow_html=True)
    st.markdown('<div class="aiopf-title">AI‑OPF</div>', unsafe_allow_html=True)
    st.markdown('<p class="aiopf-sub">From grid simulation to AI-driven optimal dispatch. '
                'An open-source research platform combining power-system optimization, '
                'reinforcement learning, and explainable AI.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div style="padding-top:2.2rem">'
                f'<span class="aiopf-badge">{GRID_NAME}</span>'
                f'<span class="aiopf-badge">pandapower + scikit-learn</span>'
                f'</div>', unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# SIDEBAR — REPRODUCIBLE TRAINING CONFIG
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Experiment Configuration")
seed = st.sidebar.number_input("Random Seed", value=42, step=1)
n_scenarios = st.sidebar.slider("Number of Scenarios", 20, 1000, 150, step=10,
                                 help="Each scenario runs one real OPF solve (pandapower/PYPOWER). "
                                      "Larger values take longer in a CPU-only environment.")
n_estimators = st.sidebar.slider("RF n_estimators", 50, 500, 200, step=50)
max_depth = st.sidebar.slider("RF max_depth", 4, 30, 12, step=1)

st.sidebar.caption("Algorithm: supervised RandomForest AI-OPF surrogate "
                    "(RL / PPO stage is scaffolded in aiopf_core.py for future work).")

run_clicked = st.sidebar.button("▶ START TRAINING", use_container_width=True)
reproduce_clicked = st.sidebar.button("↻ REPRODUCE LAST EXPERIMENT", use_container_width=True)

if "last_config" not in st.session_state:
    st.session_state.last_config = None
if "run_data" not in st.session_state:
    st.session_state.run_data = None

def do_run(seed, n_scenarios, n_estimators, max_depth):
    progress = st.progress(0.0, text="Starting...")
    status = st.empty()

    def cb(frac, msg):
        progress.progress(min(frac, 1.0), text=msg)

    t0 = time.time()
    out = run_experiment(seed=int(seed), n_scenarios=int(n_scenarios),
                          n_estimators=int(n_estimators), max_depth=int(max_depth),
                          progress_cb=cb)
    progress.empty()
    status.success(f"Experiment complete in {time.time()-t0:.1f}s — "
                    f"{out['results']['scenarios_solved']} scenarios solved, "
                    f"{out['results']['scenarios_failed_to_converge']} failed to converge.")
    st.session_state.run_data = out
    st.session_state.last_config = dict(seed=seed, n_scenarios=n_scenarios,
                                         n_estimators=n_estimators, max_depth=max_depth)

if run_clicked:
    do_run(seed, n_scenarios, n_estimators, max_depth)
elif reproduce_clicked and st.session_state.last_config:
    do_run(**st.session_state.last_config)
elif st.session_state.run_data is None:
    # First load: run a small default experiment so the dashboard isn't empty
    do_run(42, 60, 200, 12)

data = st.session_state.run_data
res = data["results"] if data else None

# ---------------------------------------------------------------------------
# TRAINING DASHBOARD
# ---------------------------------------------------------------------------
tab_dash, tab_score, tab_repro, tab_about = st.tabs(
    ["📊 Training Dashboard", "🏆 Scorecard", "🔁 Reproducibility Center", "ℹ️ About & Credits"])

with tab_dash:
    if res:
        ds = res["dataset_sizes"]
        cols = st.columns(3)
        cols[0].markdown(f'<div class="metric-card"><div class="metric-label">Training</div>'
                          f'<div class="metric-value">{ds["training_samples"]}</div></div>', unsafe_allow_html=True)
        cols[1].markdown(f'<div class="metric-card"><div class="metric-label">Validation</div>'
                          f'<div class="metric-value">{ds["validation_samples"]}</div></div>', unsafe_allow_html=True)
        cols[2].markdown(f'<div class="metric-card"><div class="metric-label">Testing</div>'
                          f'<div class="metric-value">{ds["test_samples"]}</div></div>', unsafe_allow_html=True)

        st.write("")
        m = res["metrics_measured"]
        mc = st.columns(4)
        mc[0].markdown(f'<div class="metric-card"><div class="metric-label">Feasibility Rate</div>'
                        f'<div class="metric-value">{m["feasibility_rate_pct"]}%</div></div>', unsafe_allow_html=True)
        mc[1].markdown(f'<div class="metric-card"><div class="metric-label">Mean Optimality Gap</div>'
                        f'<div class="metric-value">{m["mean_optimality_gap_pct"]}%</div></div>', unsafe_allow_html=True)
        mc[2].markdown(f'<div class="metric-card"><div class="metric-label">Constraint Violations</div>'
                        f'<div class="metric-value">{m["constraint_violations_count"]}/{ds["test_samples"]}</div></div>', unsafe_allow_html=True)
        mc[3].markdown(f'<div class="metric-card"><div class="metric-label">Inference Time</div>'
                        f'<div class="metric-value">{m["inference_time_ms_per_sample"]} ms</div></div>', unsafe_allow_html=True)

        st.write("")
        fig = go.Figure(go.Bar(
            x=[m["feasibility_rate_pct"], m["mean_optimality_gap_pct"],
               m["median_optimality_gap_pct"], m["constraint_violation_rate_pct"]],
            y=["Feasibility Rate", "Mean Opt. Gap", "Median Opt. Gap", "Violation Rate"],
            orientation="h",
            marker_color=["#34D399", "#F59E0B", "#F59E0B", "#F87171"],
        ))
        fig.update_layout(paper_bgcolor="#0D1420", plot_bgcolor="#0D1420",
                           font_color="#CBD5E1", height=280,
                           margin=dict(l=10, r=10, t=20, b=10),
                           xaxis=dict(gridcolor="#1E293B", ticksuffix="%"))
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Stage 1 (this run): supervised RandomForest surrogate trained on verified "
                   "pandapower/PYPOWER OPF solutions. Stage 2 (Gymnasium + Stable-Baselines3 PPO) "
                   "is scaffolded for future work — no numbers are shown for it since it hasn't been trained.")
    else:
        st.info("Click **START TRAINING** in the sidebar to run a real experiment.")

with tab_score:
    if res:
        m, c = res["metrics_measured"], res["cost_stats_eur"]
        rows = [
            ("Feasibility Rate", f'{m["feasibility_rate_pct"]}%'),
            ("Mean Optimality Gap", f'{m["mean_optimality_gap_pct"]}%'),
            ("Median Optimality Gap", f'{m["median_optimality_gap_pct"]}%'),
            ("Constraint Violation Rate", f'{m["constraint_violation_rate_pct"]}%'),
            ("Constraint Violations", f'{m["constraint_violations_count"]}'),
            ("Inference Time", f'{m["inference_time_ms_per_sample"]} ms'),
            ("Training Time", f'{m["training_time_seconds"]} s'),
            ("Scenario Generation Time", f'{m["scenario_generation_time_seconds"]} s'),
            ("Mean True OPF Cost", f'€{c["true_mean_cost"]}' if c["true_mean_cost"] else "—"),
            ("Mean Surrogate Cost", f'€{c["predicted_mean_cost"]}' if c["predicted_mean_cost"] else "—"),
        ]
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        for k, v in rows:
            st.markdown(f'<div class="scorecard-row"><span style="color:#64748B">{k}</span><span>{v}</span></div>',
                        unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.caption(f"Grid: {res['grid']} · Algorithm: {res['algorithm']} · Seed: {res['seed']} · "
                   f"Scenarios solved: {res['scenarios_solved']} (failed: {res['scenarios_failed_to_converge']})")
    else:
        st.info("Run an experiment to see the scorecard.")

with tab_repro:
    if res and data:
        st.subheader("Experiment Configuration")
        st.json(res, expanded=False)

        colA, colB, colC = st.columns(3)
        colA.download_button("⬇ experiment.json", data=json.dumps(res, indent=2),
                              file_name="experiment.json", mime="application/json",
                              use_container_width=True)

        csv_buf = io.StringIO()
        data["dataframe"].to_csv(csv_buf, index=False)
        colB.download_button("⬇ dataset.csv", data=csv_buf.getvalue(),
                              file_name="dataset.csv", mime="text/csv",
                              use_container_width=True)

        model_buf = io.BytesIO()
        joblib.dump(data["model"], model_buf)
        colC.download_button("⬇ AI-OPF-RF-v1.joblib", data=model_buf.getvalue(),
                              file_name="AI-OPF-RF-v1.joblib",
                              mime="application/octet-stream",
                              use_container_width=True)

        st.caption("Re-running with the same seed and scenario count reproduces this dataset "
                   "and model exactly (pandapower OPF is deterministic given the same inputs).")
    else:
        st.info("Run an experiment to unlock exports.")

with tab_about:
    st.markdown("#### Open-Source Technologies")
    stack = ["pandapower", "PYPOWER", "NumPy", "SciPy", "scikit-learn",
             "PyTorch (planned RL)", "Gymnasium (planned RL)", "Stable-Baselines3 (planned RL)",
             "Streamlit", "Plotly", "Pandas"]
    st.markdown(" ".join(f'<span class="aiopf-badge">{s}</span>' for s in stack), unsafe_allow_html=True)
    st.caption("Third-party libraries and the IEEE 14-bus benchmark remain the property of their "
               "respective authors under their own licenses. This project claims ownership only of "
               "the AI-OPF architecture, experiments, and results built on top of them.")

    st.markdown("#### About")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Purpose**\n\n- Academic Research\n- M.Tech Project\n- Power-System Optimization\n"
                "- AI-Based Energy Management\n- Renewable Integration")
    c2.markdown("**Combines**\n\n- Power-System Optimization\n- Artificial Intelligence\n"
                "- Reinforcement Learning\n- Digital Twins\n- Explainable AI")
    c3.markdown("**Status**\n\n- ✅ Grid simulation\n- ✅ Synthetic scenarios\n- ✅ Conventional OPF ground truth\n"
                "- ✅ Supervised AI-OPF surrogate\n- ⬜ RL policy (PPO) — planned\n- ⬜ SHAP explainability — planned")

st.markdown('<div class="aiopf-footer">AI-OPF · Intelligent Power Grid Optimization<br/>'
            'Open-Source AI &amp; Power-System Research Platform<br/><br/>'
            '<span class="name">Designed &amp; Developed by Kaifi</span><br/>'
            '© 2026 Kaifi. All Rights Reserved.</div>', unsafe_allow_html=True)
