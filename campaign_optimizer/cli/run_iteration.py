from __future__ import annotations

import argparse

from campaign_optimizer.capillary.batch import build_candidate_batch
from campaign_optimizer.capillary.plotting import write_basic_plots
from campaign_optimizer.config import load_optimizer_config
from campaign_optimizer.io import read_table, write_json
from campaign_optimizer.objectives import build_objectives
from campaign_optimizer.observations import build_observations
from campaign_optimizer.recommend import propose_recommendations
from campaign_optimizer.reporting import build_report
from campaign_optimizer.state import write_optimizer_state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument(
        "--build-candidate-batch",
        action="store_true",
        help=(
            "After writing recommended_candidates.tsv, also build "
            "outputs/candidate_batch.tsv and outputs/batch_campaign_plan.json."
        ),
    )
    parser.add_argument(
        "--build-report",
        action="store_true",
        help=(
            "After writing optimizer artifacts, also build lightweight reports "
            "and plots under iter_XXX/reports and iter_XXX/plots."
        ),
    )
    args = parser.parse_args(argv)

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

    report_paths = None
    if args.build_report:
        report_paths = build_report(cfg, args.iteration)

    candidate_batch_path = None
    batch_campaign_plan_path = None
    if args.build_candidate_batch:
        candidate_batch_path, batch_campaign_plan_path = build_candidate_batch(
            cfg,
            args.iteration,
        )

    print(f"[OK] observations     {observations}")
    print(f"[OK] objectives       {objectives}")
    print(f"[OK] recommendations  {recommendations}")
    print(f"[OK] state            {state}")

    if args.build_candidate_batch:
        print(f"[OK] candidate_batch      {candidate_batch_path}")
        print(f"[OK] batch_campaign_plan  {batch_campaign_plan_path}")

    if report_paths is not None:
        print(f"[OK] report_manifest      {report_paths['plot_manifest']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
