"""ChemPrior training entrypoint."""

import json
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
    if config.run.accelerator == "auto":
        config.run.accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    if config.run.devices == "auto":
        config.run.devices = (
            1 if config.run.accelerator == "cpu" else torch.cuda.device_count()
        )
    L.seed_everything(config.run.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    module = ChemPriorTrainingModule(config)
    datamodule = TrainDataModule(config)
    batches_per_epoch = datamodule.batches_per_epoch
    steps_per_epoch = batches_per_epoch // config.accumulate_grad_batches
    save_steps = round(steps_per_epoch * config.run.save_every_n_epochs)
    val_steps = round(steps_per_epoch * config.run.val_every_n_epochs)
    destination = Path(config.run.save_dir)
    checkpoint_dir = destination / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = (
        WandbLogger(
            project=config.run.wandb_project,
            group=config.train.stage,
            save_dir=str(destination),
        )
        if config.run.use_wandb
        else CSVLogger(
            str(destination),
            name="metrics",
            flush_logs_every_n_steps=config.run.log_every_n_steps,
        )
    )
    periodic_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="checkpoint-{step:09d}",
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
                filename="step={step}-val={val_loss:.4f}",
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
        accumulate_grad_batches=config.accumulate_grad_batches,
        gradient_clip_val=config.train.gradient_clip_val,
        use_distributed_sampler=False,
        check_val_every_n_epoch=None,
        val_check_interval=val_steps * config.accumulate_grad_batches,
        log_every_n_steps=config.run.log_every_n_steps,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        limit_val_batches=1.0 if config.dataset.val_split else 0,
        callbacks=callbacks,
        logger=logger,
    )
    if trainer.is_global_zero:
        (destination / "config.json").write_text(
            json.dumps(config.to_dict(), indent=2) + "\n"
        )
    trainer.fit(module, datamodule=datamodule, ckpt_path=config.run.resume_checkpoint)
    if config.dataset.val_split:
        trainer.validate(module, datamodule=datamodule)
    trainer.save_checkpoint(destination / "last.ckpt")
    if trainer.is_global_zero:
        module.model.save_pretrained(destination / "model.pt")
    return trainer, module
