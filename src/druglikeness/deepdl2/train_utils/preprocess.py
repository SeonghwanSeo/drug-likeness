"""SMILES to compressed Parquet, then local Hugging Face Arrow datasets."""

import re
from collections.abc import Iterable, Iterator
from io import TextIOWrapper
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Union

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

SMILES_SCHEMA: pa.Schema = pa.schema([("smiles", pa.string())])
ATOM_MAPPING = re.compile(r"\[[^\]]*:\d+\]")


def _smiles_rows(
    paths: list[str],
    smiles_column: int,
    canonical: bool,
) -> Iterator[dict[str, str]]:
    if canonical:
        from ..model import canonicalize

    for path in paths:
        with (
            pa.input_stream(path) as stream,
            TextIOWrapper(stream, encoding="utf-8") as source,
        ):
            for line in source:
                smiles = line.split()[smiles_column]
                if canonical:
                    smiles = canonicalize(smiles)
                if ATOM_MAPPING.search(smiles):
                    continue
                yield {"smiles": smiles}


def smiles_to_parquet(
    paths: Iterable[Union[str, Path]],
    output_dir: Union[str, Path],
    *,
    smiles_column: int = 0,
    canonical: bool = True,
    compression_level: int = 5,
    max_shard_size: int = 500_000_000,
) -> dict[str, int]:
    """Write Parquet shards, rotating at the compressed size target after each group."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True)
    rows = _smiles_rows([str(path) for path in paths], smiles_column, canonical)
    count = size = shards = 0
    batch = list(islice(rows, 100_000))
    while batch:
        output_path = output_dir / f"shard-{shards:05d}.parquet"
        with (
            pa.OSFile(str(output_path), "wb") as sink,
            pq.ParquetWriter(
                sink,
                SMILES_SCHEMA,
                compression="zstd",
                compression_level=compression_level,
                use_dictionary=False,
            ) as writer,
        ):
            while batch:
                writer.write_table(pa.Table.from_pylist(batch, schema=SMILES_SCHEMA))
                count += len(batch)
                batch = list(islice(rows, 100_000))
                if sink.tell() >= max_shard_size:
                    break
        size += output_path.stat().st_size
        shards += 1
    return {"rows": count, "bytes": size, "shards": shards}


def parquet_to_arrow(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    num_shards: Optional[int] = None,
    num_proc: Optional[int] = None,
) -> dict[str, int]:
    """Decompress Parquet into a memory-mapped HF dataset; do not tokenize."""
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir.parent) as cache:
        dataset = load_dataset(
            "parquet",
            data_files=[str(path) for path in sorted(Path(input_dir).glob("*.parquet"))],
            columns=["smiles"],
            split="train",
            cache_dir=cache,
            num_proc=num_proc,
        )
        dataset.save_to_disk(str(output_dir), num_shards=num_shards, num_proc=num_proc)
        return {"rows": len(dataset)}
