from __future__ import annotations

import argparse

from campaign_optimizer.config import load_optimizer_config
from campaign_optimizer.recommend import propose_recommendations
from campaign_optimizer.state import write_optimizer_state
from campaign_optimizer.capillary.plotting import write_basic_plots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    args = parser.parse_args()

    cfg = load_optimizer_config(args.config)
    iter_dir = cfg.iteration_dir(args.iteration)

    path = propose_recommendations(cfg, args.iteration)
    write_basic_plots(iter_dir)

    write_optimizer_state(
        iter_dir=iter_dir,
        iteration=args.iteration,
        config=cfg.data,
        objective_config=cfg.objective_config(),
        parameter_space=cfg.parameter_space(),
        observations_path=iter_dir / "inputs" / "observations.csv",
        objectives_path=iter_dir / "inputs" / "objective_table.csv",
        recommendations_path=path,
        surrogate_summary_path=iter_dir / "outputs" / "surrogate_summary.json",
    )

    print(f"[OK] wrote {path}")


if __name__ == "__main__":
    main()
