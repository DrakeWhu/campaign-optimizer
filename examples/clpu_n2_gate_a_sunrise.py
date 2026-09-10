from __future__ import annotations

import argparse
import json
from pathlib import Path

from campaign_optimizer.capillary.gate_a import (
    prepare_gate_a_staging,
    validate_gate_a,
)


def _defaults() -> dict[str, Path]:
    runs = Path.home() / "warpx_runs"
    reference = runs / "clpu_capillary_guiding_baseline_soft50_trfix_v2" / "optimizer_runs" / "iter_020" / "inputs"
    return {
        "warpx_runs_root": runs,
        "reference_manifest": reference / "source_campaigns_effective.json",
        "reference_observations": reference / "observations.csv",
        "reference_objectives": reference / "objective_table.csv",
        "directed_scan_root": runs / "clpu_f32_radius_density_a0_7case_20260826",
    }


def main() -> int:
    defaults = _defaults()
    parser = argparse.ArgumentParser(
        description=(
            "Prepare or validate the non-destructive CLPU N2 344-observation Gate A. "
            "This command never launches WarpX or submits jobs."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--staging-root", required=True)
    prepare.add_argument("--warpx-runs-root", default=str(defaults["warpx_runs_root"]))
    prepare.add_argument("--reference-manifest", default=str(defaults["reference_manifest"]))
    prepare.add_argument("--reference-observations", default=str(defaults["reference_observations"]))
    prepare.add_argument("--reference-objectives", default=str(defaults["reference_objectives"]))
    prepare.add_argument("--directed-scan-root", default=str(defaults["directed_scan_root"]))

    validate = sub.add_parser("validate")
    validate_group = validate.add_mutually_exclusive_group(required=True)
    validate_group.add_argument("--staging-root")
    validate_group.add_argument("--config")
    validate.add_argument("--bootstrap-receipt")

    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_gate_a_staging(
            staging_root=args.staging_root,
            reference_manifest_path=args.reference_manifest,
            reference_observations_path=args.reference_observations,
            reference_objectives_path=args.reference_objectives,
            directed_scan_root=args.directed_scan_root,
            warpx_runs_root=args.warpx_runs_root,
        )
    else:
        config = (
            Path(args.config)
            if args.config
            else Path(args.staging_root) / "optimizer.json"
        )
        result = validate_gate_a(
            config_path=config,
            bootstrap_receipt_path=args.bootstrap_receipt,
        )

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
