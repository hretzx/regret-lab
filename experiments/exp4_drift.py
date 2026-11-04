import numpy as np
from src.metrics import cli, print_table, run_scenario
from src.plotstyle import plot_curves, save_fig

def main():
    args = cli(trials=300, horizon=3000)
    res = run_scenario("drift", args.horizon, args.trials, args.seed)
    x = np.arange(1, args.horizon + 1)
    change = args.horizon // 3
    save_fig(plot_curves({k: v["regret"] for k, v in res.items()}, x,
                         "patients treated", "cumulative regret", vline=change), "exp4_drift_regret")
    save_fig(plot_curves({k: v["best"] for k, v in res.items()}, x,
                         "patients treated", "% on best arm", vline=change), "exp4_drift_best")
    print(f"[drift] change at patient {change} trials={args.trials} seed={args.seed}")
    print_table(res)

if __name__ == "__main__":
    main()
