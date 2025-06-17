from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class DeepDLTrainConfig:
    pretrained_model: str = "chemsci-2021-pretrain"

    # datamodule
    train_data_path: str = "data/train/worlddrug_not_fda.smi"
    val_data_path: Optional[str] = None
    split_ratio: float = 0.9  # use if val_data_path is None
    num_workers: int = 4
    batch_size: int = 256

    # optimizer
    lr: float = 1e-4
    weight_decay: float = 0.01

    # scheduler (cosine scheduler with warmup)
    warmup_ratio: float = 0.1

    # trainer
    seed: int = 123
    max_epochs: int = 100
    num_gpus: int = 1
    gradient_clip_val: float = 1.0
    precision: int | str = 32

    # logger
    save_dir: str = "./result"
    use_wandb: bool = False
    log_every_n_steps: int = 10

    # checkpointing
    check_val_every_n_epoch: int = 10
    checkpoint_epochs: int = 10
    monitor: str = "val_score"
    monitor_mode: str = "max"

    def to_dict(self) -> dict:
        return asdict(self)
