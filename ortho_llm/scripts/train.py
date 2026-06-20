from __future__ import annotations

import argparse

from ortho_llm.config import load_config
from ortho_llm.train import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Orthogonal LLaMA pretraining")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional dotlist overrides, e.g. train.num_steps=100 optim.default_role_optimizer=so",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    train(config)


if __name__ == "__main__":
    main()
