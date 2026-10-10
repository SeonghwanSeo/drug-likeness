"""SMILES to compressed Parquet, then local Hugging Face Arrow datasets."""

import json
import math
import re
from collections.abc import Iterable, Iterator
from io import TextIOWrapper
from itertools import islice
from multiprocessing import get_context
from pathlib import Path
from typing import Optional, Union
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import DatasetInfo, Features, Value
from datasets.arrow_writer import ArrowWriter
from datasets.utils import tqdm

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


def _write_arrow_shard(
    job: tuple[str, list[tuple[str, int, int, int]]],
) -> int:
    path, segments = job
    with ArrowWriter(
        path=path,
        features=Features({"smiles": Value("string")}),
        writer_batch_size=100_000,
    ) as writer:
        for source, row_group, offset, length in segments:
            with pq.ParquetFile(source) as parquet:
                table = parquet.read_row_group(
                    row_group, columns=["smiles"], use_threads=False
                )
            writer.write_table(table.slice(offset, length))
        rows, _ = writer.finalize()
    return rows


def parquet_to_arrow(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    num_shards: Optional[int] = None,
    num_proc: Optional[int] = None,
) -> dict[str, int]:
    """Stream Parquet row groups directly into final HF Arrow shards."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True)
    groups = []
    total_rows = total_bytes = 0
    paths = sorted(Path(input_dir).glob("*.parquet"))
    for path in tqdm(paths, desc="Reading Parquet metadata", unit="file"):
        with pq.ParquetFile(path) as parquet:
            column = parquet.schema_arrow.names.index("smiles")
            for i in range(parquet.num_row_groups):
                group = parquet.metadata.row_group(i)
                if group.num_rows:
                    groups.append((str(path), i, group.num_rows))
                total_rows += group.num_rows
                total_bytes += group.column(column).total_uncompressed_size

    if num_shards is None:
        num_shards = max(1, math.ceil(total_bytes / 500_000_000))
    filenames = [f"data-{i:05d}-of-{num_shards:05d}.arrow" for i in range(num_shards)]
    jobs = []
    group_index = offset = 0
    for i, filename in enumerate(filenames):
        remaining = total_rows // num_shards + (i < total_rows % num_shards)
        segments = []
        while remaining:
            source, row_group, rows = groups[group_index]
            length = min(remaining, rows - offset)
            segments.append((source, row_group, offset, length))
            remaining -= length
            offset += length
            if offset == rows:
                group_index += 1
                offset = 0
        jobs.append((str(output_dir / filename), segments))

    with (
        get_context("spawn").Pool(processes=num_proc or 1) as pool,
        tqdm(total=total_rows, desc="Writing HF Arrow", unit="rows") as progress,
    ):
        for rows in pool.imap_unordered(_write_arrow_shard, jobs):
            progress.update(rows)

    DatasetInfo(features=Features({"smiles": Value("string")})).write_to_directory(
        str(output_dir)
    )
    # HF save_to_disk metadata; publish only after all final shards are written.
    state = {
        "_data_files": [{"filename": name} for name in filenames],
        "_fingerprint": uuid4().hex[:16],
        "_format_columns": None,
        "_format_kwargs": {},
        "_format_type": None,
        "_output_all_columns": False,
        "_split": "train",
    }
    (output_dir / "state.json").write_text(json.dumps(state, indent=2) + "\n")
    return {"rows": total_rows}
