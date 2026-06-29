from __future__ import annotations

import argparse

from campaign_optimizer.config import load_optimizer_config
from campaign_optimizer.observations import build_observations
from campaign_optimizer.objectives import build_objectives
from campaign_optimizer.recommend import propose_recommendations
from campaign_optimizer.io import read_table, write_json
from campaign_optimizer.state import write_optimizer_state
from campaign_optimizer.capillary.plotting import write_basic_plots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    args = parser.parse_args()

    cfg = load_optimizer_config(args.config)
    iter_dir = cfg.iteration_dir(args.iteration)

    observations = build_observations(cfg, args.iteration)
    objectives = build_objectives(cfg, args.iteration)

    obj = read_table(objectives)
    fit = obj[obj["fit_eligible"].astype(str).str.lower() == "true"]

    surrogate_summary = write_json(
        iter_dir / "outputs" / "surrogate_summary.json",
        {
            "schema_version": 1,
            "backend": "passive_nearest_observed",
            "status": "ok",
            "fit_rows": int(len(fit)),
            "note": (
                "Phase 1 passive baseline. Optimas/AxModelManager can replace "
                "this backend without changing file contracts."
            ),
        },
    )

    recommendations = propose_recommendations(cfg, args.iteration)
    write_basic_plots(iter_dir)

    state = write_optimizer_state(
        iter_dir=iter_dir,
        iteration=args.iteration,
        config=cfg.data,
        objective_config=cfg.objective_config(),
        parameter_space=cfg.parameter_space(),
        observations_path=observations,
        objectives_path=objectives,
        recommendations_path=recommendations,
        surrogate_summary_path=surrogate_summary,
    )

    print(f"[OK] observations     {observations}")
    print(f"[OK] objectives       {objectives}")
    print(f"[OK] recommendations  {recommendations}")
    print(f"[OK] state            {state}")


if __name__ == "__main__":
    main()
