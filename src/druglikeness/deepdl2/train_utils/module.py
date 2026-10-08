"""Lightning module: objective, optimizer, data loaders and resumable state."""

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import lightning as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..model import DeepDL2, DeepDL2Config
from ..tokenizer import SmilesTokenizer
from .dataset import ArrowBatches, read_index


@dataclass
class DeepDL2TrainConfig:
    train_index: str
    save_dir: str
    val_index: Optional[str] = None
    model: DeepDL2Config = field(default_factory=DeepDL2Config)
    stage: str = "pretrain"
    init_checkpoint: Optional[str] = None  # weights only, fresh optimizer/schedule
    resume_checkpoint: Optional[str] = None  # full state and data cursor
    batch_size: int = 256  # per device
    max_length: int = 255  # SMILES tokens; padded input/target width is this + 1
    accumulate_grad_batches: int = 1
    num_workers: int = 4
    max_steps: int = 10000  # optimizer steps, including warmup and decay
    warmup_steps: int = 500
    decay_steps: int = 1000  # final cosine decay; 0 keeps the stable LR
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_clip_val: float = 1.0
    precision: str = "bf16-mixed"
    accelerator: str = "gpu"
    devices: int = 1
    compile_model: bool = True  # Transformer only; GRU uses packed sequences
    compile_mode: str = "default"
    seed: int = 42
    save_every_n_steps: int = 10000  # optimizer updates, not microbatches
    val_every_n_steps: int = 1000
    log_every_n_steps: int = 50
    use_wandb: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DeepDL2TrainingModule(L.LightningModule):
    def __init__(self, config: DeepDL2TrainConfig) -> None:
        super().__init__()
        self.config: DeepDL2TrainConfig = config
        self.train_index = read_index(config.train_index)
        self.batches_per_data_epoch: int = sum(
            block["rows"] // config.batch_size for block in self.train_index["blocks"]
        )
        self.val_index = read_index(config.val_index) if config.val_index else None
        tokenizer = SmilesTokenizer.from_file(
            Path(config.train_index).with_name("vocab.txt")
        )
        source = config.init_checkpoint or config.resume_checkpoint
        if source:
            self.model = DeepDL2.from_pretrained(source)
        else:
            self.model = DeepDL2.from_config(config.model, tokenizer)
        self.model.train()
        self.config.model = self.model.config
        self.consumed_batches: int = (
            0  # per rank, includes gradient accumulation microbatches
        )
        self.prediction_tokens: int = (
            0  # global across ranks; EOS included, PAD/BOS excluded
        )
        self.molecules_seen: int = 0
        self._window_stats = None
        self.save_hyperparameters(config.to_dict())

    def forward(
        self, seq: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.model(seq, lengths)

    def loss(
        self,
        seq: torch.Tensor,
        logits: torch.Tensor,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ) -> torch.Tensor:
        return F.cross_entropy(
            self.model.prediction_logits(logits).flatten(0, 1),
            seq.flatten(),
            ignore_index=self.model.pad_token_id,
            reduction=reduction,
        )

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        # Decay matrix weights, not biases or LayerNorm parameters; tied weights
        # appear only once in parameters().
        decay, no_decay = [], []
        for parameter in self.model.parameters():
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)
        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.config.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.config.lr,
        )
        c = self.config

        def schedule(step: int) -> float:
            if step < c.warmup_steps:
                return (step + 1) / c.warmup_steps
            decay_start = c.max_steps - c.decay_steps
            if c.decay_steps == 0 or step < decay_start:
                return 1.0
            progress = (step - decay_start) / c.decay_steps
            return c.min_lr_ratio + (1 - c.min_lr_ratio) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_fit_start(self) -> None:
        self._window_stats = torch.zeros(3, dtype=torch.float64, device=self.device)
        if self.config.compile_model and self.model.config.architecture == "transformer":
            # In-place compilation preserves state_dict keys and model exports.
            self.model.compile(dynamic=False, mode=self.config.compile_mode)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        # Dropout randomness depends on consumed data, not worker prefetch or how
        # many validation/checkpoint calls happened before a resume.
        if self.model.config.architecture == "gru":
            torch.manual_seed(
                self.config.seed
                + self.consumed_batches * self.trainer.world_size
                + self.global_rank
            )
        seq, lengths = batch["seq"], batch["lengths"]
        loss_sum = self.loss(seq, self(seq, lengths), reduction="sum")
        self._window_stats[0] += loss_sum.detach().double()
        self._window_stats[1] += lengths.sum()
        self._window_stats[2] += seq.size(0)
        self.consumed_batches = batch["cursor"]
        # Lightning divides by accumulate_grad_batches, and DDP averages ranks.
        # on_before_optimizer_step normalizes the entire accumulation window.
        # The fixed batch-size divisor is removed when the token count is known.
        return loss_sum / self.config.batch_size

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        stats = self.trainer.strategy.reduce(self._window_stats.clone(), reduce_op="sum")
        factor = (
            self.trainer.world_size
            * self.config.accumulate_grad_batches
            * self.config.batch_size
            / stats[1]
        )
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(factor.to(parameter.grad.dtype))
        self.prediction_tokens += int(stats[1])
        self.molecules_seen += int(stats[2])
        self.log("train_loss", (stats[0] / stats[1]).float(), on_step=True, prog_bar=True)
        self.log("prediction_tokens", float(self.prediction_tokens), on_step=True)
        self.log("molecules_seen", float(self.molecules_seen), on_step=True)
        self.log(
            "data_epoch",
            self.consumed_batches * self.trainer.world_size / self.batches_per_data_epoch,
            on_step=True,
            prog_bar=True,
        )
        self._window_stats.zero_()

    def on_validation_epoch_start(self) -> None:
        self._validation_stats = torch.zeros(3, dtype=torch.float64, device=self.device)

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        seq, lengths = batch["seq"], batch["lengths"]
        self._validation_stats[0] += self.loss(
            seq, self(seq, lengths), reduction="sum"
        ).double()
        self._validation_stats[1] += lengths.sum()
        self._validation_stats[2] += seq.size(0)

    def on_validation_epoch_end(self) -> None:
        # Ranks can have different validation batch counts: one collective only
        # after the full shard.
        stats = self.trainer.strategy.reduce(
            self._validation_stats.clone(), reduce_op="sum"
        )
        self.log(
            "val_loss", (stats[0] / stats[1]).float(), sync_dist=False, prog_bar=True
        )
        self.log("val_log_likelihood", (-stats[0] / stats[2]).float(), sync_dist=False)

    def _loader(self, training: bool) -> DataLoader:
        index = self.train_index if training else self.val_index
        # Lightning restores its fetched-batch counter on resume. Keep the full
        # horizon as the loader length; only offset the dataset's starting cursor.
        total_batches = self.config.max_steps * self.config.accumulate_grad_batches
        dataset = ArrowBatches(
            index,
            self.config.batch_size,
            seed=self.config.seed,
            rank=self.global_rank,
            world_size=self.trainer.world_size,
            training=training,
            start_batch=self.consumed_batches if training else 0,
            num_batches=total_batches if training else None,
            tokenizer=self.model.tokenizer,
            max_length=self.config.max_length,
        )
        return DataLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=self.config.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(True)

    def val_dataloader(self) -> Optional[DataLoader]:
        return self._loader(False) if self.val_index is not None else None

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["config"] = self.model.config.to_dict()
        checkpoint["tokens"] = list(self.model.tokenizer.tokens)
        checkpoint["consumed_batches"] = self.consumed_batches
        checkpoint["prediction_tokens"] = self.prediction_tokens
        checkpoint["molecules_seen"] = self.molecules_seen

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.consumed_batches = checkpoint["consumed_batches"]
        self.prediction_tokens = checkpoint["prediction_tokens"]
        self.molecules_seen = checkpoint["molecules_seen"]
