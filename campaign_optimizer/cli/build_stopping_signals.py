from __future__ import annotations

import argparse

from campaign_optimizer.config import load_optimizer_config
from campaign_optimizer.reporting import build_stopping_signals


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    args = parser.parse_args(argv)

    cfg = load_optimizer_config(args.config)
    path = build_stopping_signals(cfg, args.iteration)

    print(f"[OK] stopping_signals {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
