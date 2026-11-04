from collections import deque
import numpy as np

def argmax_random_ties(values, rng):
    return int(rng.choice(np.flatnonzero(values == np.max(values))))

class Policy:
    def __init__(self, n_arms, rng):
        self.k, self.rng = n_arms, rng
        self.counts = np.zeros(n_arms)
        self.sums = np.zeros(n_arms)
    def means(self):
        return self.sums / np.maximum(self.counts, 1)
    def update(self, arm, reward):
        self.counts[arm] += 1
        self.sums[arm] += reward

class EqualSplit(Policy):
    def select_arm(self, t):
        return t % self.k

class EpsilonGreedy(Policy):
    def __init__(self, n_arms, rng, epsilon=0.1):
        super().__init__(n_arms, rng)
        self.epsilon = epsilon

    def select_arm(self, t):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.k))
        return argmax_random_ties(self.means(), self.rng)

class UCB1(Policy):
    def select_arm(self, t):
        untried = np.flatnonzero(self.counts == 0)
        if len(untried):
            return int(untried[0])
        total = self.counts.sum()
        bonus = np.sqrt(2 * np.log(total) / self.counts)
        return argmax_random_ties(self.means() + bonus, self.rng)

class Thompson(Policy):
    def select_arm(self, t):
        draws = self.rng.beta(1 + self.sums, 1 + self.counts - self.sums)
        return int(np.argmax(draws))

class SlidingWindowUCB(Policy):
    def __init__(self, n_arms, rng, window=200):
        super().__init__(n_arms, rng)
        self.window = window
        self.history = deque()
        self.w_counts = np.zeros(n_arms)
        self.w_sums = np.zeros(n_arms)

    def select_arm(self, t):
        untried = np.flatnonzero(self.w_counts == 0)
        if len(untried):
            return int(untried[0])
        n = len(self.history)
        bonus = np.sqrt(2 * np.log(n) / self.w_counts)
        return argmax_random_ties(self.w_sums / self.w_counts + bonus, self.rng)

    def update(self, arm, reward):
        super().update(arm, reward)
        self.history.append((arm, reward))
        self.w_counts[arm] += 1
        self.w_sums[arm] += reward
        if len(self.history) > self.window:
            old_arm, old_reward = self.history.popleft()
            self.w_counts[old_arm] -= 1
            self.w_sums[old_arm] -= old_reward


class DiscountedUCB(Policy):
    def __init__(self, n_arms, rng, gamma=0.99):
        super().__init__(n_arms, rng)
        self.gamma = gamma
        self.d_counts = np.zeros(n_arms)
        self.d_sums = np.zeros(n_arms)

    def select_arm(self, t):
        untried = np.flatnonzero(self.counts == 0)
        if len(untried):
            return int(untried[0])
        bonus = np.sqrt(2 * np.log(self.d_counts.sum()) / self.d_counts)
        return argmax_random_ties(self.d_sums / self.d_counts + bonus, self.rng)

    def update(self, arm, reward):
        super().update(arm, reward)
        self.d_counts *= self.gamma
        self.d_sums *= self.gamma
        self.d_counts[arm] += 1
        self.d_sums[arm] += reward


ALGORITHMS = {"equal": EqualSplit, "epsilon_greedy": EpsilonGreedy, "ucb1": UCB1,
              "thompson": Thompson, "sw_ucb": SlidingWindowUCB, "d_ucb": DiscountedUCB}
MAIN_ALGOS = {"Equal split": ("equal", {}),
              "Epsilon-greedy": ("epsilon_greedy", {"epsilon": 0.1}),
              "UCB1": ("ucb1", {}),
              "Thompson": ("thompson", {})}


def drift_algos(windows=(100, 300, 1000), gamma=0.99):
    algos = {"UCB1": ("ucb1", {})}
    for w in windows:
        algos[f"SW-UCB {w}"] = ("sw_ucb", {"window": w})
    algos[f"D-UCB {gamma}"] = ("d_ucb", {"gamma": gamma})
    return algos


def make_policy(name, n_arms, rng, **params):
    return ALGORITHMS[name](n_arms, rng, **params)
