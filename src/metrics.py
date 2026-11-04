import argparse
import numpy as np
from .algorithms import MAIN_ALGOS, drift_algos, make_policy
from .environment import make_scenario

def run_trial(env, policy, horizon, rng):
    step_regret = np.zeros(horizon)
    hits = np.zeros(horizon)
    for t in range(horizon):
        arm = policy.select_arm(t)
        policy.update(arm, env.pull(arm, t, rng))
        p = env.probs_at(t)
        step_regret[t] = p.max() - p[arm]
        hits[t] = p[arm] == p.max()
    return np.cumsum(step_regret), np.cumsum(hits) / np.arange(1, horizon + 1)

def run_many(env, algo, params, horizon, n_trials, seed):
    regret = np.empty((n_trials, horizon))
    best = np.empty((n_trials, horizon))
    for i, child in enumerate(np.random.SeedSequence(seed).spawn(n_trials)):
        env_rng, algo_rng = (np.random.default_rng(s) for s in child.spawn(2))
        policy = make_policy(algo, env.n_arms, algo_rng, **params)
        regret[i], best[i] = run_trial(env, policy, horizon, env_rng)
    return regret, best

def mean_ci(x):
    mean = x.mean(axis=0)
    half = 1.96 * x.std(axis=0, ddof=1) / np.sqrt(x.shape[0])
    return mean, mean - half, mean + half

def compare(env, algos, horizon, n_trials, seed):
    out = {}
    for label, (name, params) in algos.items():
        regret, best = run_many(env, name, params, horizon, n_trials, seed)
        out[label] = {"regret": mean_ci(regret), "best": mean_ci(best * 100)}
    return out

def run_scenario(scenario, horizon, n_trials, seed):
    env = make_scenario(scenario, horizon, seed)
    algos = drift_algos() if scenario == "drift" else MAIN_ALGOS
    return compare(env, algos, horizon, n_trials, seed)

def table_rows(results):
    rows = []
    for label, r in results.items():
        (rm, _, rh), (bm, _, bh) = r["regret"], r["best"]
        rows.append((label, rm[-1], rh[-1] - rm[-1], bm[-1], bh[-1] - bm[-1]))
    return rows

def print_table(results):
    print(f"{'algorithm':<18}{'final regret (95% CI)':>26}{'% on best arm (95% CI)':>26}")
    for label, r, rc, b, bc in table_rows(results):
        print(f"{label:<18}{r:>16.1f} ± {rc:<6.1f}{b:>16.1f} ± {bc:<6.1f}")


def cli(trials=300, horizon=2000):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=trials)
    ap.add_argument("--horizon", type=int, default=horizon)
    return ap.parse_args()
