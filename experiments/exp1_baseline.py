import numpy as np
from src.metrics import cli, print_table, run_scenario
from src.plotstyle import plot_curves, save_fig

def main():
    args = cli(trials=300, horizon=2000)
    x = np.arange(1, args.horizon + 1)
    for scenario in ["large gap", "small gap", "many arms"]:
        res = run_scenario(scenario, args.horizon, args.trials, args.seed)
        tag = scenario.replace(" ", "_")
        save_fig(plot_curves({k: v["regret"] for k, v in res.items()}, x,
                             "patients treated", "cumulative regret"), f"exp1_{tag}_regret")
        save_fig(plot_curves({k: v["best"] for k, v in res.items()}, x,
                             "patients treated", "% on best arm"), f"exp1_{tag}_best")
        print(f"\\n[{scenario}] trials={args.trials} horizon={args.horizon} seed={args.seed}")
        print_table(res)

if __name__ == "__main__":
    main()
