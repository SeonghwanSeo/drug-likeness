import hashlib
import random
from typing import Optional

import lightning as L
import torch
from datasets import IterableDataset, load_dataset
from datasets.distributed import split_dataset_by_node
from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)
from torch.utils.data import DataLoader

from ..tokenizer import SmilesTokenizer
from .config import ChemPriorDatasetConfig, ChemPriorTrainConfig


def _load_smiles_dataset(
    config: "ChemPriorDatasetConfig",
    split: str,
) -> IterableDataset:
    """Load a dataset split for streaming."""
    return load_dataset(**config.load_kwargs(), split=split, streaming=True)


class SmilesCollator:
    """Tokenize SMILES, append EOS and pad each batch."""

    def __init__(
        self,
        max_length: int = 127,
        enumerate_stereo: bool = False,
        stereo_seed: Optional[int] = None,
    ) -> None:
        self.max_length: int = max_length
        self.enumerate_stereo: bool = enumerate_stereo
        self.stereo_seed: Optional[int] = stereo_seed
        self.tokenizer: SmilesTokenizer = SmilesTokenizer()

    def _sample_stereoisomer(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        rng = random
        if self.stereo_seed is not None:
            identity = int.from_bytes(
                hashlib.blake2b(smiles.encode(), digest_size=8).digest(), "big"
            )
            rng = random.Random(self.stereo_seed ^ identity)
        options = StereoEnumerationOptions(
            onlyUnassigned=True,
            unique=True,
            maxIsomers=1024,
            rand=rng.getrandbits(64),
        )
        isomers = list(EnumerateStereoisomers(mol, options=options))
        return Chem.MolToSmiles(rng.choice(isomers), canonical=True, isomericSmiles=True)

    def __call__(self, batch: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        seq = torch.full(
            (len(batch), self.max_length + 1),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
        )
        lengths = torch.empty(len(batch), dtype=torch.long)
        for i, row in enumerate(batch):
            smiles = row["smiles"]
            if self.enumerate_stereo:
                smiles = self._sample_stereoisomer(smiles)
            tokens = self.tokenizer.encode(smiles, add_special_tokens=False)
            tokens = tokens[: self.max_length]
            tokens.append(self.tokenizer.eos_token_id)
            seq[i, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            lengths[i] = len(tokens)
        return {"seq": seq, "lengths": lengths}


class TrainDataModule(L.LightningDataModule):
    """Build training and validation loaders from Parquet streams."""

    def __init__(self, config: "ChemPriorTrainConfig") -> None:
        super().__init__()
        self.config: ChemPriorTrainConfig = config
        self.train_dataset = _load_smiles_dataset(
            config.dataset, config.dataset.train_split
        )
        self.train_rows = self.train_dataset.info.splits[
            config.dataset.train_split
        ].num_examples
        self.val_dataset: Optional[IterableDataset] = None
        if config.dataset.val_split:
            self.val_dataset = _load_smiles_dataset(
                config.dataset, config.dataset.val_split
            )

    @property
    def batches_per_epoch(self) -> int:
        c = self.config
        """Number of per-device batches in an epoch of complete optimizer updates."""
        return (
            self.train_rows
            // (c.train.batch_size * c.run.devices * c.accumulate_grad_batches)
            * c.accumulate_grad_batches
        )

    def train_dataloader(self) -> DataLoader:
        c = self.config
        dataset = self.train_dataset.shuffle(
            seed=c.run.seed + self.trainer.current_epoch,
            buffer_size=c.dataset.shuffle_buffer,
        )
        dataset = split_dataset_by_node(
            dataset, rank=self.trainer.global_rank, world_size=self.trainer.world_size
        ).repeat(None)
        # Repeat to supply the configured batch count on every rank.
        return DataLoader(
            dataset,
            batch_size=c.train.batch_size,
            collate_fn=SmilesCollator(
                c.dataset.max_length, enumerate_stereo=c.dataset.enumerate_stereo
            ),
            num_workers=c.dataset.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None:
            return None
        c = self.config
        dataset = split_dataset_by_node(
            self.val_dataset,
            rank=self.trainer.global_rank,
            world_size=self.trainer.world_size,
        )
        return DataLoader(
            dataset,
            batch_size=c.train.batch_size,
            collate_fn=SmilesCollator(
                c.dataset.max_length,
                enumerate_stereo=c.dataset.enumerate_stereo,
                stereo_seed=c.run.seed,
            ),
            num_workers=c.dataset.num_workers,
            drop_last=True,
            pin_memory=True,
        )
