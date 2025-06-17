import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch.optim import AdamW

from druglikeness.deepdl.model import DeepDL
from druglikeness.deepdl.vocab import PAD_TOKEN_ID

from .config import DeepDLTrainConfig
from .utils import get_cosine_schedule_with_warmup


class DeepDLTrainingModule(L.LightningModule):
    def __init__(self, config: DeepDLTrainConfig):
        super().__init__()
        self.config = config
        self.model: DeepDL = self.construct_model()
        self.pad_token_id: int = PAD_TOKEN_ID
        self.save_hyperparameters(config.to_dict())

    def construct_model(self) -> DeepDL:
        return DeepDL()

    def on_save_checkpoint(self, checkpoint: dict):
        new_state_dict = {}
        for k, v in checkpoint["state_dict"].items():
            assert k.startswith("model.")
            new_state_dict[k.removeprefix("model.")] = v
        checkpoint["state_dict"] = new_state_dict

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        return OptimizerLRSchedulerConfig(
            optimizer=optimizer,
            lr_scheduler={
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        )

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """return logits"""
        return self.model.forward(input_ids, lengths)  # [B, L, C]

    def scoring(self, input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """return scores"""
        p_x = self.model.calc_p_x(input_ids, logits)
        # normalize to [0, 100]
        return (p_x + 100).clamp(min=0)

    def loss_fn(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.flatten(end_dim=-2)  # [B*L, C]
        target = target.flatten()  # [B*L]
        return F.cross_entropy(pred, target, ignore_index=self.pad_token_id)

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        input_ids, lengths = batch  # [B, L], [B,]
        logits = self(input_ids, lengths)  # [B, L, C]
        loss = self.loss_fn(logits, input_ids)  # [B, L, C] vs [B, L]
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        input_ids, lengths = batch  # [B, L], [B,]
        logits = self(input_ids, lengths)  # [B, L, C]
        loss = self.loss_fn(logits, input_ids)  # [B, L, C] vs [B, L]

        # scoring
        scores = self.scoring(input_ids, logits)
        avg_score = scores.mean()

        self.log_dict(
            {"val_loss": loss, "val_score": avg_score},
            sync_dist=True,
            prog_bar=True,
        )


class DeepDLFinetuningModule(DeepDLTrainingModule):
    def construct_model(self) -> DeepDL:
        model = DeepDL.from_pretrained(self.config.pretrained_model, "cpu")
        model.train()
        return model

    def configure_optimizers(self):
        # use smaller learning rate for embedding and GRU layers
        lr_embed = self.config.lr * 0.5
        lr_gru = self.config.lr * 0.5
        lr_head = self.config.lr

        parameter_groups = [
            {"params": self.model.embedding.parameters(), "lr": lr_embed},
            {"params": self.model.GRU.parameters(), "lr": lr_gru},
            {"params": self.model.fc.parameters(), "lr": lr_head},
        ]

        optimizer = AdamW(parameter_groups, lr=self.config.lr, weight_decay=self.config.weight_decay)

        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        return OptimizerLRSchedulerConfig(
            optimizer=optimizer,
            lr_scheduler={
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        )
