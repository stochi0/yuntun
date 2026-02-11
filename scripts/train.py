#!/usr/bin/env python3
"""Train Qwen3-style model on FineWeb dataset."""

import argparse
from pathlib import Path

# Add src to path for direct run
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.yuntun.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train on FineWeb")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.json",
        help="Path to training config JSON",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu, auto-detected if not set)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max_steps from config (for quick testing)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).parent.parent / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")

    trainer = Trainer(config_path=str(config_path), device=args.device)
    if args.max_steps is not None:
        trainer.max_steps = args.max_steps
    trainer.train()


if __name__ == "__main__":
    main()
