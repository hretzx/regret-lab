import numpy as np
from src.algorithms import MAIN_ALGOS
from src.environment import BernoulliBandit
from src.metrics import cli, compare
from src.plotstyle import plot_curves, save_fig

GAPS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
BEST = 0.6

def main():
    args = cli(trials=300, horizon=1000)
    final = {label: ([], [], []) for label in MAIN_ALGOS}
    for gap in GAPS:
        res = compare(BernoulliBandit([BEST, BEST - gap]), MAIN_ALGOS,
                      args.horizon, args.trials, args.seed)
        for label, r in res.items():
            for store, series in zip(final[label], r["regret"]):
                store.append(series[-1])
    curves = {k: tuple(np.array(a) for a in v) for k, v in final.items()}
    save_fig(plot_curves(curves, np.array(GAPS), "gap between best and second-best arm",
                         f"regret after {args.horizon} patients"), "exp3_gap_sensitivity")
    print(f"{'gap':<8}" + "".join(f"{k:>16}" for k in curves))
    for i, gap in enumerate(GAPS):
        print(f"{gap:<8.2f}" + "".join(f"{curves[k][0][i]:>16.1f}" for k in curves))

if __name__ == "__main__":
    main()
