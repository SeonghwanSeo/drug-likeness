"""HF SMILES datasets, online tokenization and fixed-width Lightning loaders."""

from typing import TYPE_CHECKING, Optional

import lightning as L
import torch
from datasets import Dataset, load_from_disk
from datasets.distributed import split_dataset_by_node
from torch.utils.data import DataLoader

from ..tokenizer import SmilesTokenizer

if TYPE_CHECKING:
    from .module import DeepDL2TrainConfig


class SmilesCollator:
    def __init__(self, max_length: int = 127) -> None:
        self.max_length: int = max_length
        self.tokenizer: SmilesTokenizer = SmilesTokenizer()

    def __call__(self, batch: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        seq = torch.full(
            (len(batch), self.max_length + 1),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
        )
        lengths = torch.empty(len(batch), dtype=torch.long)
        for i, row in enumerate(batch):
            # Keep a lexical prefix, then EOS; the model inserts BOS internally.
            tokens = self.tokenizer.encode(row["smiles"], add_special_tokens=False)
            tokens = tokens[: self.max_length]
            tokens.append(self.tokenizer.eos_token_id)
            seq[i, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            lengths[i] = len(tokens)
        return {"seq": seq, "lengths": lengths}


class TrainDataModule(L.LightningDataModule):
    def __init__(self, config: "DeepDL2TrainConfig") -> None:
        super().__init__()
        self.config: DeepDL2TrainConfig = config
        self.train_dataset: Dataset = load_from_disk(config.train_data)
        self.val_dataset: Optional[Dataset] = (
            load_from_disk(config.val_data) if config.val_data else None
        )

    @property
    def batches_per_epoch(self) -> int:
        c = self.config
        # Whole optimizer updates; an epoch is approximately one corpus pass.
        return (
            len(self.train_dataset)
            // (c.batch_size * c.devices * c.accumulate_grad_batches)
            * c.accumulate_grad_batches
        )

    def train_dataloader(self) -> DataLoader:
        c = self.config
        dataset = self.train_dataset.to_iterable_dataset(
            num_shards=len(self.train_dataset.cache_files)
        ).shuffle(seed=c.seed + self.trainer.current_epoch, buffer_size=c.shuffle_buffer)
        dataset = split_dataset_by_node(
            dataset, rank=self.trainer.global_rank, world_size=self.trainer.world_size
        ).repeat(None)
        # Repeating avoids unequal shard lengths ending one DDP rank early.
        # Trainer limits each epoch to the same number of batches on every rank.
        return DataLoader(
            dataset,
            batch_size=c.batch_size,
            collate_fn=SmilesCollator(c.max_length),
            num_workers=c.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None:
            return None
        c = self.config
        dataset = self.val_dataset.to_iterable_dataset(
            num_shards=len(self.val_dataset.cache_files)
        )
        dataset = split_dataset_by_node(
            dataset, rank=self.trainer.global_rank, world_size=self.trainer.world_size
        )
        return DataLoader(
            dataset,
            batch_size=c.batch_size,
            collate_fn=SmilesCollator(c.max_length),
            num_workers=c.num_workers,
            drop_last=True,
            pin_memory=True,
        )
