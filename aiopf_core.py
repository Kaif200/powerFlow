"""
Core AI-OPF pipeline logic, refactored as a reusable function so the
Streamlit app can trigger real runs interactively.
"""
import time
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

GRID_NAME = "IEEE 14-Bus"


def build_base_net():
    return pn.case14()


def run_experiment(seed=42, n_scenarios=150, n_estimators=200, max_depth=12,
                    progress_cb=None):
    """
    Runs the full AI-OPF pipeline: scenario generation, conventional OPF
    ground truth, ML surrogate training, and held-out evaluation.

    progress_cb: optional callable(fraction: float, message: str) for UI progress bars.
    Returns a dict of real, measured results plus the raw dataframe/model.
    """
    rng = np.random.default_rng(seed)
    base_net = build_base_net()
    n_load = len(base_net.load)
    n_gen = len(base_net.gen)
    base_loads_p = base_net.load['p_mw'].values.copy()
    base_loads_q = base_net.load['q_mvar'].values.copy()

    records = []
    failed = 0
    t_start = time.time()

    for i in range(n_scenarios):
        net = build_base_net()
        load_scale = rng.uniform(0.7, 1.3, size=n_load)
        net.load['p_mw'] = base_loads_p * load_scale
        net.load['q_mvar'] = base_loads_q * load_scale

        renewable_factor = rng.uniform(0.4, 1.0, size=n_gen)
        net.gen['max_p_mw'] = base_net.gen['max_p_mw'].values * (0.6 + 0.4 * renewable_factor)

        if rng.random() < 0.15:
            derate_idx = rng.integers(0, n_gen)
            net.gen.loc[derate_idx, 'max_p_mw'] *= 0.3

        try:
            pp.runopp(net, verbose=False, numba=False)
            converged = net["OPF_converged"]
        except Exception:
            converged = False

        if not converged:
            failed += 1
        else:
            feat = {
                **{f"load_p_{j}": net.load['p_mw'].values[j] for j in range(n_load)},
                **{f"load_q_{j}": net.load['q_mvar'].values[j] for j in range(n_load)},
                **{f"gen_max_{j}": net.gen['max_p_mw'].values[j] for j in range(n_gen)},
            }
            target = {
                **{f"gen_p_{j}": net.res_gen['p_mw'].values[j] for j in range(n_gen)},
                "cost": float(net.res_cost),
            }
            records.append({**feat, **target})

        if progress_cb and (i + 1) % max(1, n_scenarios // 20) == 0:
            progress_cb((i + 1) / n_scenarios, f"Solving OPF scenario {i+1}/{n_scenarios}")

    gen_time = time.time() - t_start
    df = pd.DataFrame(records)
    feature_cols = [c for c in df.columns if c.startswith("load_") or c.startswith("gen_max_")]
    target_cols = [c for c in df.columns if c.startswith("gen_p_")]

    X = df[feature_cols].values
    Y = df[target_cols].values
    COST = df["cost"].values

    X_train, X_temp, Y_train, Y_temp, C_train, C_temp = train_test_split(
        X, Y, COST, test_size=0.4, random_state=seed)
    X_val, X_test, Y_val, Y_test, C_val, C_test = train_test_split(
        X_temp, Y_temp, C_temp, test_size=0.5, random_state=seed)

    if progress_cb:
        progress_cb(0.9, "Training ML surrogate model")

    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                   random_state=seed, n_jobs=-1)
    t0 = time.time()
    model.fit(X_train, Y_train)
    train_time = time.time() - t0

    t0 = time.time()
    Y_pred = model.predict(X_test) if len(X_test) else np.zeros((0, n_gen))
    infer_time_total = time.time() - t0
    inference_ms = (infer_time_total / max(1, len(X_test))) * 1000

    poly = base_net.poly_cost.set_index('element')

    def approx_cost(p_dispatch):
        total = 0.0
        for j in range(n_gen):
            row = poly.loc[j]
            cp0 = row.get('cp0_eur', 0.0)
            cp1 = row.get('cp1_eur_per_mw', 0.0)
            cp2 = row.get('cp2_eur_per_mw2', 0.0)
            p = max(p_dispatch[j], 0.0)
            total += cp0 + cp1 * p + cp2 * p ** 2
        return total

    pred_costs = np.array([approx_cost(Y_pred[i]) for i in range(len(Y_pred))]) if len(Y_pred) else np.array([])
    true_gen_costs = np.array([approx_cost(Y_test[i]) for i in range(len(Y_test))]) if len(Y_test) else np.array([])

    if len(true_gen_costs):
        gap = np.abs(pred_costs - true_gen_costs) / np.clip(np.abs(true_gen_costs), 1e-6, None) * 100
        mean_gap, median_gap = float(np.mean(gap)), float(np.median(gap))
    else:
        mean_gap = median_gap = float("nan")

    if len(Y_pred):
        max_p_test = X_test[:, [feature_cols.index(f"gen_max_{j}") for j in range(n_gen)]]
        within = np.all((Y_pred >= -1e-3) & (Y_pred <= max_p_test + 1e-3), axis=1)
        feasibility_rate = float(np.mean(within) * 100)
        violation_count = int(np.sum(~within))
        violation_rate = float(violation_count / len(Y_pred) * 100)
    else:
        feasibility_rate = violation_count = violation_rate = float("nan")

    if progress_cb:
        progress_cb(1.0, "Done")

    results = {
        "grid": GRID_NAME,
        "algorithm": "RandomForestRegressor (scikit-learn) — supervised AI-OPF surrogate",
        "seed": seed,
        "scenarios_requested": n_scenarios,
        "scenarios_solved": len(records),
        "scenarios_failed_to_converge": failed,
        "dataset_sizes": {
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
            "test_samples": len(X_test),
        },
        "metrics_measured": {
            "feasibility_rate_pct": round(feasibility_rate, 3) if feasibility_rate == feasibility_rate else None,
            "mean_optimality_gap_pct": round(mean_gap, 3) if mean_gap == mean_gap else None,
            "median_optimality_gap_pct": round(median_gap, 3) if median_gap == median_gap else None,
            "constraint_violation_rate_pct": round(violation_rate, 3) if violation_rate == violation_rate else None,
            "constraint_violations_count": violation_count if violation_count == violation_count else None,
            "inference_time_ms_per_sample": round(inference_ms, 4),
            "training_time_seconds": round(train_time, 3),
            "scenario_generation_time_seconds": round(gen_time, 2),
        },
        "cost_stats_eur": {
            "true_mean_cost": round(float(np.mean(true_gen_costs)), 3) if len(true_gen_costs) else None,
            "predicted_mean_cost": round(float(np.mean(pred_costs)), 3) if len(pred_costs) else None,
        },
        "hyperparameters": {"n_estimators": n_estimators, "max_depth": max_depth},
    }

    return {
        "results": results,
        "dataframe": df,
        "model": model,
        "feature_cols": feature_cols,
        "target_cols": target_cols,
    }
