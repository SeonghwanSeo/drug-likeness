import argparse
from pathlib import Path

from druglikeness.deepdl.train_utils.train import DeepDLTrainConfig, train_deepdl


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a model with specified dataset.")

    # pretrained model
    parser.add_argument("-a", "--arch", type=str, default="deepdl", help="Architecture of the model to be fine-tuned.")
    parser.add_argument(
        "--pretrained_model", type=str, default="chemsci-2021-pretrain", help="Name or Path to the pretrained model."
    )

    # logging
    parser.add_argument("-n", "--name", type=str, help="Prefix. If None, use the file name of `--data_path`.")
    parser.add_argument("--root_dir", type=str, default="result", help="Root directory to save the results.")

    # dataset and dataloader
    parser.add_argument(
        "--data_path", type=Path, default="./data/train/worlddrug_not_fda.smi", help="Dataset (.smi) path."
    )
    parser.add_argument("--split_ratio", type=float, default=0.9, help="Train/Val split ratio.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training.")

    # optimizer and scheduler
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for the optimizer.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Ratio of warmup steps to total steps.")

    # trainer
    parser.add_argument("--epoch", type=int, default=100, help="Number of epochs for training.")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use for training.")
    parser.add_argument("--precision", type=str, default="32", help="Precision")
    parser.add_argument("--log_interval", type=int, default=1, help="Number of steps for logging.")
    parser.add_argument("--val_interval", type=int, default=10, help="Number of epochs for validation.")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="Number of epochs for checkpointing.")
    parser.add_argument("--wandb", action="store_true", help="Turn on wandb logger.")

    args = parser.parse_args()

    name = args.name if args.name else Path(args.data_path).stem
    save_dir = Path(args.root_dir) / args.arch / name

    config = DeepDLTrainConfig(
        save_dir=save_dir,
        pretrained_model=args.pretrained_model,
        train_data_path=args.data_path,
        split_ratio=args.split_ratio,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_epochs=args.epoch,
        num_gpus=args.num_gpus,
        precision=args.precision,
        use_wandb=args.wandb,
        check_val_every_n_epoch=args.val_interval,
        checkpoint_epochs=args.checkpoint_interval,
        log_every_n_steps=args.log_interval,
    )
    return config


if __name__ == "__main__":
    config = parse_args()
    train_deepdl(config)
