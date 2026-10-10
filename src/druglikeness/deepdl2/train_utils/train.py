"""Lightning entrypoint for pretraining, continuation and fine-tuning."""

import json
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy

from .dataset import TrainDataModule
from .module import DeepDL2TrainConfig, DeepDL2TrainingModule


def train_deepdl2(config: DeepDL2TrainConfig) -> tuple[L.Trainer, DeepDL2TrainingModule]:
    L.seed_everything(config.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    module = DeepDL2TrainingModule(config)
    datamodule = TrainDataModule(config)
    destination = Path(config.save_dir)
    checkpoint_dir = destination / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = (
        WandbLogger(project="druglikeness", group=config.stage, save_dir=str(destination))
        if config.use_wandb
        else CSVLogger(
            str(destination),
            name="metrics",
            flush_logs_every_n_steps=config.log_every_n_steps,
        )
    )
    periodic_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="checkpoint-{step:09d}",
        auto_insert_metric_name=False,
        every_n_train_steps=config.save_every_n_steps,
        save_top_k=-1,
        save_last=True,
        save_on_train_epoch_end=False,
    )
    callbacks: list[Callback] = [
        periodic_checkpoint,
        LearningRateMonitor(logging_interval="step"),
    ]
    if config.val_data:
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_dir / "best",
                monitor="val_loss",
                mode="min",
                filename="step={step}-val={val_loss:.4f}",
                auto_insert_metric_name=False,
                save_top_k=3,
                save_on_train_epoch_end=False,
            )
        )
    trainer = L.Trainer(
        default_root_dir=destination,
        accelerator=config.accelerator,
        devices=config.devices,
        strategy=DDPStrategy(broadcast_buffers=False) if config.devices > 1 else "auto",
        max_epochs=config.max_epochs,
        limit_train_batches=datamodule.batches_per_epoch,
        reload_dataloaders_every_n_epochs=1,
        precision=config.precision,
        accumulate_grad_batches=config.accumulate_grad_batches,
        gradient_clip_val=config.gradient_clip_val,
        use_distributed_sampler=False,
        check_val_every_n_epoch=None,
        val_check_interval=config.val_every_n_steps * config.accumulate_grad_batches,
        log_every_n_steps=config.log_every_n_steps,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        limit_val_batches=1.0 if config.val_data else 0,
        callbacks=callbacks,
        logger=logger,
    )
    if trainer.is_global_zero:
        (destination / "config.json").write_text(
            json.dumps(config.to_dict(), indent=2) + "\n"
        )
    trainer.fit(module, datamodule=datamodule, ckpt_path=config.resume_checkpoint)
    if config.val_data:
        trainer.validate(module, datamodule=datamodule)
    trainer.save_checkpoint(destination / "last.ckpt")
    if trainer.is_global_zero:
        module.model.save_pretrained(destination / "model.pt")
    return trainer, module
