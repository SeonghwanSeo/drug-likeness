from dataclasses import asdict, dataclass
from typing import Any, Optional, Union

from ..model import ChemPriorConfig


@dataclass
class ChemPriorDatasetConfig:
    """Dataset source, splits and tokenization settings."""

    path: str
    name: Optional[str] = None
    data_files: Optional[Union[str, list[str], dict[str, Union[str, list[str]]]]] = None
    data_dir: Optional[str] = None
    train_split: str = "train"
    val_split: Optional[str] = None

    max_length: int = 127
    enumerate_stereo: bool = False
    num_workers: int = 4
    shuffle_buffer: int = 100000

    def load_kwargs(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "name": self.name,
            "data_files": self.data_files,
            "data_dir": self.data_dir,
        }


@dataclass
class ChemPriorOptimizationConfig:
    model: ChemPriorConfig
    stage: str = "pretrain"
    pretrained_model: Optional[str] = None
    batch_size: int = 512
    max_epochs: int = 10
    warmup_steps: int = 2000
    decay_steps: int = 0
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_clip_val: float = 1.0
    global_batch_size: int = 16384


@dataclass
class ChemPriorRunConfig:
    save_dir: str
    resume_checkpoint: Optional[str] = None
    precision: str = "bf16-mixed"
    accelerator: str = "auto"
    devices: Union[int, str] = "auto"
    compile_model: bool = True
    compile_mode: str = "default"
    seed: int = 42
    save_every_n_epochs: float = 0.1
    val_every_n_epochs: float = 1
    log_every_n_steps: int = 100
    use_wandb: bool = False
    wandb_project: str = "chemprior"


@dataclass
class ChemPriorTrainConfig:
    dataset: ChemPriorDatasetConfig
    run: ChemPriorRunConfig
    train: ChemPriorOptimizationConfig

    @classmethod
    def from_dict(cls, settings: dict[str, Any]) -> "ChemPriorTrainConfig":
        training = dict(settings["train"])
        model = training["model"]
        training["model"] = (
            ChemPriorConfig.preset(model)
            if isinstance(model, str)
            else ChemPriorConfig(**model)
        )
        return cls(
            dataset=ChemPriorDatasetConfig(**settings["dataset"]),
            train=ChemPriorOptimizationConfig(**training),
            run=ChemPriorRunConfig(**settings["run"]),
        )

    @property
    def accumulate_grad_batches(self) -> int:
        return self.train.global_batch_size // (self.train.batch_size * self.run.devices)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
