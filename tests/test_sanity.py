import numpy as np
from src.algorithms import ALGORITHMS, make_policy
from src.environment import BernoulliBandit
from src.metrics import mean_ci, run_many, run_trial

def _trial(name, probs, horizon=300, seed=0, **params):
    env = BernoulliBandit(probs)
    policy = make_policy(name, env.n_arms, np.random.default_rng(seed + 1), **params)
    regret, best = run_trial(env, policy, horizon, np.random.default_rng(seed))
    return policy, regret, best


def test_single_arm_has_zero_regret():
    for name in ALGORITHMS:
        assert np.all(_trial(name, [0.5])[1] == 0)


def test_ucb_tries_every_arm():
    policy, _, _ = _trial("ucb1", [0.1, 0.2, 0.3, 0.4, 0.5], horizon=50)
    assert np.all(policy.counts >= 1)


def test_regret_nonnegative_nondecreasing_and_share_in_range():
    for name in ALGORITHMS:
        _, regret, best = _trial(name, [0.2, 0.5, 0.8])
        assert regret[0] >= 0 and np.all(np.diff(regret) >= 0)
        assert np.all((best >= 0) & (best <= 1))


def test_same_seed_gives_same_result():
    env = BernoulliBandit([0.3, 0.6])
    a = run_many(env, "thompson", {}, 100, 5, seed=3)[0]
    b = run_many(env, "thompson", {}, 100, 5, seed=3)[0]
    assert np.array_equal(a, b)


def test_thompson_beats_equal_split_on_easy_problem():
    env = BernoulliBandit([0.2, 0.5, 0.8])
    equal = run_many(env, "equal", {}, 300, 20, seed=0)[0][:, -1].mean()
    thompson = run_many(env, "thompson", {}, 300, 20, seed=0)[0][:, -1].mean()
    assert thompson < equal


def test_confidence_interval_brackets_mean():
    m, lo, hi = mean_ci(np.random.default_rng(0).normal(size=(50, 4)))
    assert np.all(lo < m) and np.all(m < hi)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
