from __future__ import annotations

import argparse

from campaign_optimizer.config import load_optimizer_config
from campaign_optimizer.objectives import build_objectives


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    args = parser.parse_args()

    cfg = load_optimizer_config(args.config)
    path = build_objectives(cfg, args.iteration)
    print(f"[OK] wrote {path}")


if __name__ == "__main__":
    main()
