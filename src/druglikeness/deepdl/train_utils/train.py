from pathlib import Path
from typing import Optional

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers.wandb import WandbLogger
from torch.utils.data import DataLoader

from .config import DeepDLTrainConfig
from .dataset import SmilesDataset, collate_fn
from .module import DeepDLFinetuningModule


def build_trainer(config: DeepDLTrainConfig) -> L.Trainer:
    summary = ModelSummary(max_depth=2)
    checkpointing = ModelCheckpoint(
        filename="{epoch}-{step}-{val_score:.2f}",
        every_n_epochs=config.checkpoint_epochs,
        monitor=config.monitor,
        mode=config.monitor_mode,
        save_last=True,
        save_top_k=3,
    )
    lr_monitor = LearningRateMonitor("step")
    early_stopping = EarlyStopping(monitor=config.monitor, mode=config.monitor_mode, verbose=False, min_delta=0.02)
    callbacks: list[Callback] = [summary, checkpointing, lr_monitor, early_stopping]

    if config.use_wandb:
        logger = WandbLogger(
            name=Path(config.save_dir).stem,
            project="druglikeness",
            group="finetune",
            log_model=True,
            save_dir=config.save_dir,
        )
    else:
        logger = None

    trainer = L.Trainer(
        default_root_dir=config.save_dir,
        devices=config.num_gpus,
        max_epochs=config.max_epochs,
        log_every_n_steps=config.log_every_n_steps,
        gradient_clip_val=config.gradient_clip_val,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        precision=config.precision,
        logger=logger,
        callbacks=callbacks,
    )
    return trainer


def build_dataloaders(config: DeepDLTrainConfig) -> tuple[DataLoader, DataLoader]:
    train_set = SmilesDataset.from_file(config.train_data_path)
    if config.val_data_path is not None:
        valid_set = SmilesDataset.from_file(config.val_data_path)
    elif config.split_ratio < 1.0:
        # train-test split
        train_set_size = int(len(train_set) * config.split_ratio)
        valid_set_size = len(train_set) - train_set_size
        generator = torch.Generator().manual_seed(config.seed)
        train_set, valid_set = torch.utils.data.random_split(train_set, [train_set_size, valid_set_size], generator)
    else:
        # use the same dataset for training and validation
        valid_set = train_set

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=True,
    )
    return train_loader, valid_loader


def train_deepdl(config: DeepDLTrainConfig):
    torch.set_float32_matmul_precision("high")
    L.seed_everything(config.seed, workers=True)

    # create save directory
    Path(config.save_dir).mkdir(parents=True, exist_ok=True)

    # build trainer
    trainer = build_trainer(config)

    # build dataloaders
    train_loader, valid_loader = build_dataloaders(config)

    # construct model
    model = DeepDLFinetuningModule(config)

    trainer.fit(model, train_loader, valid_loader)
