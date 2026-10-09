"""Lightning training objective, optimizer and WSD schedule."""

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import lightning as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch.nn import functional as F

from ..model import DeepDL2, DeepDL2Config


@dataclass
class DeepDL2TrainConfig:
    train_data: str
    save_dir: str
    val_data: Optional[str] = None
    model: DeepDL2Config = field(default_factory=DeepDL2Config)
    stage: str = "pretrain"
    init_checkpoint: Optional[str] = None
    resume_checkpoint: Optional[str] = None  # model/optimizer/loop; no data cursor
    batch_size: int = 256  # per device, before gradient accumulation
    max_length: int = 127  # SMILES tokens; input width includes one BOS/EOS slot
    accumulate_grad_batches: int = 1
    num_workers: int = 4
    shuffle_buffer: int = 10000
    max_epochs: int = 5
    max_steps: int = -1
    warmup_steps: int = 500
    decay_steps: int = 0  # final cosine decay; 0 keeps the stable LR
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_clip_val: float = 1.0
    precision: str = "bf16-mixed"
    accelerator: str = "gpu"
    devices: int = 1
    compile_model: bool = True
    compile_mode: str = "default"
    seed: int = 42
    save_every_n_steps: int = 10000
    val_every_n_steps: int = 1000
    log_every_n_steps: int = 100
    use_wandb: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DeepDL2TrainingModule(L.LightningModule):
    def __init__(self, config: DeepDL2TrainConfig) -> None:
        super().__init__()
        self.config: DeepDL2TrainConfig = config
        source = config.init_checkpoint or config.resume_checkpoint
        self.model: DeepDL2 = (
            DeepDL2.from_pretrained(source)
            if source
            else DeepDL2.from_config(config.model)
        )
        self.model.train()
        self.config.model = self.model.config
        self.save_hyperparameters(config.to_dict())

    def forward(
        self, seq: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.model(seq, lengths)

    def loss(self, seq: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            self.model.prediction_logits(logits).flatten(0, 1),
            seq.flatten(),
            ignore_index=self.model.pad_token_id,
        )

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            betas=self.config.betas,
            eps=self.config.eps,
            weight_decay=self.config.weight_decay,
        )
        c = self.config
        total_steps = self.trainer.estimated_stepping_batches

        def schedule(step: int) -> float:
            if step < c.warmup_steps:
                return (step + 1) / c.warmup_steps
            decay_start = total_steps - c.decay_steps
            if c.decay_steps == 0 or step < decay_start:
                return 1.0
            progress = min((step - decay_start) / c.decay_steps, 1.0)
            return c.min_lr_ratio + (1 - c.min_lr_ratio) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, schedule),
                "interval": "step",
            },
        }

    def on_fit_start(self) -> None:
        if self.config.compile_model and self.model.config.architecture == "transformer":
            self.model.compile(dynamic=False, mode=self.config.compile_mode)

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        seq = batch["seq"]
        loss = self.loss(seq, self(seq, batch["lengths"]))
        self.log("train_loss", loss, on_step=True, prog_bar=True, sync_dist=False)
        return loss

    @torch.no_grad()
    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        if (self.global_step + 1) % self.config.log_every_n_steps != 0:
            return
        if not self.trainer.is_global_zero:
            return
        parameters = list(self.model.parameters())
        grad_norm = torch.linalg.vector_norm(
            torch.stack(
                [
                    torch.linalg.vector_norm(p.grad.detach().float())
                    for p in parameters
                    if p.grad is not None
                ]
            )
        )
        param_norm = torch.linalg.vector_norm(
            torch.stack(
                [torch.linalg.vector_norm(p.detach().float()) for p in parameters]
            )
        )
        self.log_dict(
            {"grad_norm": grad_norm, "param_norm": param_norm},
            on_step=True,
            on_epoch=False,
            sync_dist=False,
            rank_zero_only=True,
        )

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        seq = batch["seq"]
        loss = self.loss(seq, self(seq, batch["lengths"]))
        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            sync_dist=True,
            batch_size=seq.size(0),
            prog_bar=True,
        )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["config"] = self.model.config.to_dict()
