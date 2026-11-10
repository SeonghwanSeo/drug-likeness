"""ChemPrior training entrypoint."""

import json
import math
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy

from .config import ChemPriorTrainConfig
from .dataset import TrainDataModule
from .module import ChemPriorTrainingModule


def train_chemprior(
    config: ChemPriorTrainConfig,
) -> tuple[L.Trainer, ChemPriorTrainingModule]:
    """Train a model and save its checkpoints, configuration and weights."""
    destination = Path(config.run.save_dir)
    new_pretrain = config.train.stage == "pretrain" and not config.run.resume_checkpoint
    if new_pretrain and any(
        (destination / name).exists()
        for name in ("config.json", "model.pth", "last.ckpt", "checkpoints/last.ckpt")
    ):
        raise FileExistsError(
            f"Pretrain artifacts already exist in {destination}. "
            "Choose a different run.save_dir or set run.resume_checkpoint."
        )
    if config.run.accelerator == "auto":
        config.run.accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    if config.run.devices == "auto":
        config.run.devices = (
            1 if config.run.accelerator == "cpu" else torch.cuda.device_count()
        )
    accumulation = config.accumulate_grad_batches
    for interval in (config.run.save_every_n_epochs, config.run.val_every_n_epochs):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError(
                "Checkpoint and validation epoch intervals must be positive."
            )
    L.seed_everything(config.run.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    module = ChemPriorTrainingModule(config)
    datamodule = TrainDataModule(config)
    batches_per_epoch = datamodule.batches_per_epoch
    steps_per_epoch = batches_per_epoch // accumulation
    if steps_per_epoch == 0:
        raise ValueError("The training split must contain at least one global batch.")
    save_steps = max(1, round(steps_per_epoch * config.run.save_every_n_epochs))
    val_steps = max(1, round(steps_per_epoch * config.run.val_every_n_epochs))
    checkpoint_dir = destination / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    loggers = [
        CSVLogger(
            str(destination),
            name="metrics",
            flush_logs_every_n_steps=config.run.log_every_n_steps,
        )
    ]
    if config.run.use_wandb:
        loggers.append(
            WandbLogger(
                project=config.run.wandb_project,
                name=config.run.wandb_name,
                group=config.train.stage,
                save_dir=str(destination),
            )
        )
    periodic_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch-{epoch}-step-{step}",
        auto_insert_metric_name=False,
        every_n_train_steps=save_steps,
        save_top_k=-1,
        save_last=True,
        save_on_train_epoch_end=False,
    )
    callbacks: list[Callback] = [
        periodic_checkpoint,
        LearningRateMonitor(logging_interval="step"),
    ]
    if config.dataset.val_split:
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_dir / "best",
                monitor="val_loss",
                mode="min",
                filename="step-{step}-val_loss-{val_loss:.4f}",
                auto_insert_metric_name=False,
                save_top_k=3,
                save_on_train_epoch_end=False,
            )
        )
    trainer = L.Trainer(
        default_root_dir=destination,
        accelerator=config.run.accelerator,
        devices=config.run.devices,
        strategy=DDPStrategy(broadcast_buffers=False)
        if config.run.devices > 1
        else "auto",
        max_epochs=config.train.max_epochs,
        limit_train_batches=batches_per_epoch,
        reload_dataloaders_every_n_epochs=1,
        precision=config.run.precision,
        accumulate_grad_batches=accumulation,
        gradient_clip_val=config.train.gradient_clip_val,
        use_distributed_sampler=False,
        check_val_every_n_epoch=None,
        val_check_interval=val_steps * accumulation,
        log_every_n_steps=config.run.log_every_n_steps,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        limit_val_batches=1.0 if config.dataset.val_split else 0,
        callbacks=callbacks,
        logger=loggers,
    )
    if trainer.is_global_zero:
        mode = "x" if new_pretrain else "w"
        with (destination / "training_config.json").open(mode) as config_file:
            config_file.write(json.dumps(config.to_dict(), indent=2) + "\n")
    trainer.fit(module, datamodule=datamodule, ckpt_path=config.run.resume_checkpoint)
    if config.dataset.val_split:
        trainer.validate(module, datamodule=datamodule)
    trainer.save_checkpoint(destination / "last.ckpt")
    if trainer.is_global_zero:
        module.model.save(destination)
    return trainer, module
