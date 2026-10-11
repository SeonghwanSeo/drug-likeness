import argparse
from pathlib import Path

import yaml

from druglikeness.chemprior.train_utils.config import ChemPriorTrainConfig
from druglikeness.chemprior.train_utils.train import train_chemprior


def parse_args() -> ChemPriorTrainConfig:
    parser = argparse.ArgumentParser(
        description="Train a ChemPrior model with Lightning."
    )
    parser.add_argument("config", type=Path, help="Training YAML config.")
    parser.add_argument("--resume_checkpoint", help="Resume Lightning training state.")
    args = parser.parse_args()

    settings = yaml.safe_load(args.config.read_text())
    config = ChemPriorTrainConfig.from_dict(settings)
    if args.resume_checkpoint is not None:
        config.run.resume_checkpoint = args.resume_checkpoint
    return config


if __name__ == "__main__":
    train_chemprior(parse_args())
