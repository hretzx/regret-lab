import numpy as np
from scipy import stats
from src.algorithms import MAIN_ALGOS
from src.environment import BernoulliBandit
from src.metrics import cli, compare
from src.plotstyle import plot_curves, save_fig
START = 100

def main():
    args = cli(trials=300, horizon=5000)
    env = BernoulliBandit([0.4, 0.5, 0.6])
    algos = {k: MAIN_ALGOS[k] for k in ["Equal split", "UCB1", "Thompson"]}
    res = compare(env, algos, args.horizon, args.trials, args.seed)
    t = np.arange(1, args.horizon + 1)
    log_t = np.log(t)
    curves = {k: v["regret"] for k, v in res.items()}
    save_fig(plot_curves(curves, t, "patients treated", "cumulative regret"), "exp2_regret_vs_t")
    fig = plot_curves(curves, log_t, "ln(patients treated)", "cumulative regret")
    ax = fig.axes[0]
    print(f"{'algorithm':<14}{'slope per ln t':>16}{'R2 vs ln t':>14}{'R2 vs t':>12}")
    for label, (mean, _, _) in curves.items():
        log_fit = stats.linregress(log_t[START:], mean[START:])
        lin_fit = stats.linregress(t[START:], mean[START:])
        ax.plot(log_t[START:], log_fit.intercept + log_fit.slope * log_t[START:],
                color="#BBBBBB", linewidth=0.8, linestyle="--")
        print(f"{label:<14}{log_fit.slope:>16.2f}{log_fit.rvalue ** 2:>14.4f}{lin_fit.rvalue ** 2:>12.4f}")
    save_fig(fig, "exp2_regret_vs_logt")
    print("\\nR2 is descriptive, not a proof.")

if __name__ == "__main__":
    main()
