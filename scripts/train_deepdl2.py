import argparse
from pathlib import Path

import torch
import yaml

from druglikeness.deepdl2.model import DeepDL2Config
from druglikeness.deepdl2.train_utils.module import DeepDL2TrainConfig
from druglikeness.deepdl2.train_utils.train import train_deepdl2


def parse_args() -> DeepDL2TrainConfig:
    parser = argparse.ArgumentParser(description="Train a DeepDL2 model with Lightning.")
    parser.add_argument("config", type=Path, help="Training YAML config.")
    parser.add_argument("--resume_checkpoint", help="Resume Lightning training state.")
    args = parser.parse_args()

    settings = yaml.safe_load(args.config.read_text())
    settings["model"] = DeepDL2Config.preset(settings.pop("model", "medium"))
    devices = settings.pop("devices", "auto")
    global_batch_size = settings.pop("global_batch_size")
    config = DeepDL2TrainConfig(**settings)
    config.devices = (
        (1 if config.accelerator == "cpu" else torch.cuda.device_count())
        if devices == "auto"
        else int(devices)
    )
    config.accumulate_grad_batches = global_batch_size // (
        config.batch_size * config.devices
    )
    if args.resume_checkpoint is not None:
        config.resume_checkpoint = args.resume_checkpoint
    return config


if __name__ == "__main__":
    train_deepdl2(parse_args())
