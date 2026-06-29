from __future__ import annotations

import argparse

from campaign_optimizer.capillary.batch import build_candidate_batch
from campaign_optimizer.config import load_optimizer_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    args = parser.parse_args()

    cfg = load_optimizer_config(args.config)
    candidate_batch_path, plan_path = build_candidate_batch(cfg, args.iteration)

    print(f"[OK] candidate_batch      {candidate_batch_path}")
    print(f"[OK] batch_campaign_plan  {plan_path}")


if __name__ == "__main__":
    main()
