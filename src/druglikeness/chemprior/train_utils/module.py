"""ChemPrior training with Lightning."""

import math
from typing import Any, Optional

import lightning as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch.nn import functional as F

from ..model import ChemPrior
from .config import ChemPriorTrainConfig


class ChemPriorTrainingModule(L.LightningModule):
    def __init__(self, config: ChemPriorTrainConfig) -> None:
        super().__init__()
        self.config: ChemPriorTrainConfig = config
        source = config.run.resume_checkpoint or config.train.pretrained_model
        self.model: ChemPrior = (
            ChemPrior.from_pretrained(source) if source else ChemPrior(config.train.model)
        )
        self.model.train()
        self.config.train.model = self.model.config
        self.save_hyperparameters(config.to_dict())

    def forward(
        self, seq: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.model(seq, lengths)

    def loss(self, seq: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Calculate mean token cross-entropy, ignoring padding."""
        model = self.model
        mask = torch.zeros(logits.size(-1), dtype=torch.bool, device=logits.device)
        mask[model.pad_token_id] = mask[model.bos_token_id] = True
        logits = logits.float().masked_fill(mask, -torch.inf)

        return F.cross_entropy(
            logits.flatten(0, 1),
            seq.flatten(),
            ignore_index=self.model.pad_token_id,
        )

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        c = self.config
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.train.lr,
            betas=self.config.train.betas,
            eps=self.config.train.eps,
            weight_decay=self.config.train.weight_decay,
        )
        total_steps = self.trainer.estimated_stepping_batches

        def schedule(step: int) -> float:
            if step < c.train.warmup_steps:
                return (step + 1) / c.train.warmup_steps
            decay_start = total_steps - c.train.decay_steps
            if c.train.decay_steps == 0 or step < decay_start:
                return 1.0
            progress = min((step - decay_start) / c.train.decay_steps, 1.0)
            return c.train.min_lr_ratio + (1 - c.train.min_lr_ratio) * 0.5 * (
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
        if self.config.run.compile_model:
            self.model.compile(dynamic=False, mode=self.config.run.compile_mode)

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        seq = batch["seq"]
        loss = self.loss(seq, self(seq, batch["lengths"]))
        self.log("train_loss", loss, on_step=True, prog_bar=True, sync_dist=False)
        return loss

    @torch.no_grad()
    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        if (self.global_step + 1) % self.config.run.log_every_n_steps != 0:
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
        logits = self(seq, batch["lengths"])
        loss = self.loss(seq, logits)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=int(batch["lengths"].sum().item()),
            prog_bar=True,
        )
        nll = -self.model.calc_p_x(seq, logits)
        self.log(
            "val_score",
            (100.0 - nll).clamp(min=0.0).mean() / 100.0,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["config"] = self.model.config.to_dict()
