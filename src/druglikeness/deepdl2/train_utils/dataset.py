"""Bounded-memory Arrow batches and deterministic, addressable training order.

Each Dataset item is a complete minibatch. The global batch ordinal determines
the epoch, shuffled record batch and shuffled rows without a billion-row index.
Ranks interleave these ordinals; workers only prefetch. A consumed-batch cursor
therefore resumes exactly, independently of worker count or prefetch depth.
"""

import json
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from ..tokenizer import SmilesTokenizer


def read_index(path: Union[str, Path]) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


class ArrowBatches(Dataset):
    def __init__(
        self,
        index: dict[str, Any],
        batch_size: int,
        *,
        seed: int = 20261008,
        rank: int = 0,
        world_size: int = 1,
        training: bool = False,
        start_batch: int = 0,
        num_batches: Optional[int] = None,
        tokenizer: Optional[SmilesTokenizer] = None,
        max_length: int = 255,
    ) -> None:
        self.index = index
        self.tokenizer = tokenizer or SmilesTokenizer()
        self.batch_size = batch_size
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.training = training
        self.start_batch = start_batch
        self.max_length = max_length
        self.block_batches = np.array(
            [b["rows"] // batch_size for b in index["blocks"]]
        )
        self.total_batches = int(self.block_batches.sum())
        self.num_batches = (
            num_batches
            if training
            else max(0, (self.total_batches - 1 - rank) // world_size + 1)
        )
        self._epoch = self._block_key = None
        self._tokens = self._row_order = None

    def __len__(self) -> int:
        return self.num_batches

    def location(self, item: int) -> tuple[int, int, np.ndarray]:
        """Return (epoch, record-batch index, row indices) for auditing/resume."""
        ordinal = (self.start_batch + item) * self.world_size + self.rank
        epoch, position = divmod(ordinal, self.total_batches)
        if self._epoch != epoch:
            self._epoch = epoch
            self._block_order = np.arange(len(self.block_batches))
            if self.training:
                np.random.default_rng([self.seed, epoch, 0]).shuffle(self._block_order)
            self._ends = np.cumsum(self.block_batches[self._block_order])
        slot = int(np.searchsorted(self._ends, position, side="right"))
        block = int(self._block_order[slot])
        local_batch = position - (int(self._ends[slot - 1]) if slot else 0)
        key = (epoch, block)
        if self._block_key != key:
            self._block_key = key
            self._tokens = None
            self._row_order = np.arange(self.index["blocks"][block]["rows"])
            if self.training:
                np.random.default_rng([self.seed, epoch, block + 1]).shuffle(
                    self._row_order
                )
        start = local_batch * self.batch_size
        return epoch, block, self._row_order[start : start + self.batch_size]

    def _read_tokens(self, block_number: int) -> Any:
        import pyarrow as pa
        from pyarrow import ipc

        block = self.index["blocks"][block_number]
        path = self.index["files"][block["file"]]["path"]
        with pa.memory_map(path, "r") as source:
            batch = ipc.open_file(source).get_batch(block["batch"])
            return batch.column(batch.schema.get_field_index("token_ids"))

    def __getitem__(self, item: int) -> dict[str, Any]:
        _, block_number, rows = self.location(item)
        if self._tokens is None:
            self._tokens = self._read_tokens(block_number)
            self._eligible_rows = np.flatnonzero(
                np.diff(self._tokens.offsets.to_numpy()) - 2 <= self.max_length
            )
        # Arrow arrays keep their backing buffers alive after closing the file.
        ordinal = (self.start_batch + item) * self.world_size + self.rank
        rng = np.random.default_rng([self.seed, ordinal, 2])

        targets: list[torch.Tensor] = []
        for i in rows:
            tokens = self._tokens[int(i)].as_py()[1:]
            # BOS has been stripped; exclude the remaining EOS from the limit.
            if len(tokens) - 1 > self.max_length:
                assert self.training, (
                    f"Validation sequence exceeds max_length={self.max_length}: "
                    f"record={block_number}, row={int(i)}, tokens={len(tokens) - 1}"
                )
                row = int(rng.choice(self._eligible_rows))
                tokens = self._tokens[row].as_py()[1:]
            targets.append(torch.tensor(tokens, dtype=torch.long))
        lengths = torch.zeros(self.batch_size, dtype=torch.long)
        # Targets include EOS; model.forward shifts BOS into the same fixed width.
        seq = torch.full(
            (self.batch_size, self.max_length + 1),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
        )
        for i, target in enumerate(targets):
            seq[i, : target.numel()] = target
            lengths[i] = target.numel()
        return {"seq": seq, "lengths": lengths, "cursor": self.start_batch + item + 1}
