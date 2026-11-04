import numpy as np

class BernoulliBandit:
    def __init__(self, probs):
        self.probs = np.asarray(probs, dtype=float)
        self.n_arms = len(self.probs)

    def probs_at(self, t):
        return self.probs

    def pull(self, arm, t, rng):
        return int(rng.random() < self.probs_at(t)[arm])

class DriftingBandit(BernoulliBandit):

    def __init__(self, probs_before, probs_after, change_point):
        super().__init__(probs_before)
        self.probs_after = np.asarray(probs_after, dtype=float)
        self.change_point = change_point

    def probs_at(self, t):
        return self.probs if t < self.change_point else self.probs_after


SCENARIOS = ["large gap", "small gap", "many arms", "drift"]


def make_scenario(name, horizon=2000, seed=0):
    if name == "large gap":
        return BernoulliBandit([0.2, 0.5, 0.8])
    if name == "small gap":
        return BernoulliBandit([0.5, 0.52, 0.54])
    if name == "many arms":
        return BernoulliBandit(np.random.default_rng(seed).uniform(0.1, 0.9, size=10))
    if name == "drift":
        return DriftingBandit([0.7, 0.5, 0.4], [0.3, 0.5, 0.4], change_point=horizon // 3)
    raise ValueError(f"unknown scenario: {name}")
