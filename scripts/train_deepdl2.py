import argparse

from druglikeness.deepdl2.model import DeepDL2Config
from druglikeness.deepdl2.train_utils.module import DeepDL2TrainConfig
from druglikeness.deepdl2.train_utils.train import train_deepdl2


def parse_args() -> DeepDL2TrainConfig:
    parser = argparse.ArgumentParser(description="Train a DeepDL2 model with Lightning.")
    parser.add_argument("--train_data", required=True, help="HF dataset directory.")
    parser.add_argument("--save_dir", required=True, help="Output directory.")
    parser.add_argument("--val_data", help="Optional HF validation dataset directory.")
    parser.add_argument(
        "--model", choices=["small", "medium", "large", "gru-medium"], default="medium"
    )
    parser.add_argument("--init_checkpoint", help="Initialize model weights.")
    parser.add_argument("--resume_checkpoint", help="Resume Lightning training state.")

    # Dataset and dataloader
    parser.add_argument("--batch_size", type=int, default=256, help="Batch per GPU.")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--max_length", type=int, default=127, help="SMILES tokens, excluding BOS/EOS."
    )
    parser.add_argument("--shuffle_buffer", type=int, default=10000)

    # Optimizer and scheduler
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--decay_steps", type=int, default=0)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)

    # Trainer and logging
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--no_compile", dest="compile_model", action="store_false")
    parser.add_argument("--compile_mode", default="default")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every_n_steps", type=int, default=10000)
    parser.add_argument("--val_every_n_steps", type=int, default=1000)
    parser.add_argument("--log_every_n_steps", type=int, default=100)
    parser.add_argument("--stage", default="pretrain")
    parser.add_argument("--wandb", dest="use_wandb", action="store_true")

    settings = vars(parser.parse_args())
    settings["model"] = DeepDL2Config.preset(settings["model"])
    return DeepDL2TrainConfig(**settings)


if __name__ == "__main__":
    config = parse_args()
    train_deepdl2(config)
