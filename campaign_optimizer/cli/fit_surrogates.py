from __future__ import annotations

import argparse

from campaign_optimizer.capillary.plotting import write_basic_plots
from campaign_optimizer.config import load_optimizer_config
from campaign_optimizer.io import read_table, write_json
from campaign_optimizer.state import write_optimizer_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    args = parser.parse_args()

    cfg = load_optimizer_config(args.config)
    iter_dir = cfg.iteration_dir(args.iteration)

    obj_path = iter_dir / "inputs" / "objective_table.csv"
    obj = read_table(obj_path)
    fit = obj[obj["fit_eligible"].astype(str).str.lower() == "true"]

    backend = str(
        cfg.recommendation_config().get("backend", "passive_nearest_observed")
    )

    summary_path = write_json(
        iter_dir / "outputs" / "surrogate_summary.json",
        {
            "schema_version": 1,
            "backend": backend,
            "surrogate_backend": backend,
            "status": "not_fitted_by_this_cli",
            "fit_rows": int(len(fit)),
            "note": (
                "Surrogate fitting is performed by propose_recommendations/run_iteration "
                "because the AxModelManager backend predicts directly on the candidate cloud."
            ),
        },
    )

    write_basic_plots(iter_dir)

    write_optimizer_state(
        iter_dir=iter_dir,
        iteration=args.iteration,
        config=cfg.data,
        objective_config=cfg.objective_config(),
        parameter_space=cfg.parameter_space(),
        observations_path=iter_dir / "inputs" / "observations.csv",
        objectives_path=obj_path,
        surrogate_summary_path=summary_path,
        backend=backend,
    )

    print(f"[OK] wrote {summary_path}")


if __name__ == "__main__":
    main()
