"""SMILES to Parquet converter."""

import re
from collections.abc import Iterable, Iterator
from io import TextIOWrapper
from itertools import islice
from pathlib import Path
from typing import Union

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem

SMILES_SCHEMA: pa.Schema = pa.schema([("id", pa.string()), ("smiles", pa.string())])
ATOM_MAPPING = re.compile(r"\[[^\]]*:\d+\]")


def _smiles_rows(
    paths: list[str],
    smiles_column: int,
    canonical: bool,
    id_column: int,
) -> Iterator[dict[str, str]]:
    for path in paths:
        with (
            pa.input_stream(path) as stream,
            TextIOWrapper(stream, encoding="utf-8") as source,
        ):
            for line in source:
                if not line.strip():
                    continue
                fields = line.rstrip("\r\n").split("\t")
                smiles = fields[smiles_column]
                identifier = fields[id_column]
                if ATOM_MAPPING.search(smiles):
                    continue
                if canonical:
                    smiles = Chem.MolToSmiles(
                        Chem.MolFromSmiles(smiles), canonical=True, isomericSmiles=True
                    )
                yield {"id": identifier, "smiles": smiles}


def smiles_to_parquet(
    paths: Iterable[Union[str, Path]],
    output_dir: Union[str, Path],
    *,
    smiles_column: int = 0,
    id_column: int = 1,
    dataset_name: str = "train",
    canonical: bool = True,
    compression_level: int = 5,
    max_shard_size: int = 500_000_000,
) -> dict[str, int]:
    """Convert tab-separated SMILES and IDs to Zstd-compressed Parquet files.

    Returns:
        Total row count, compressed bytes and number of files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True)
    input_paths = [str(path) for path in paths]
    rows = _smiles_rows(input_paths, smiles_column, canonical, id_column)
    count = size = shards = 0
    batch = list(islice(rows, 100_000))
    while batch:
        output_path = output_dir / f"{dataset_name}-{shards:05d}.parquet.partial"
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
    for index in range(shards):
        filename = (
            f"{dataset_name}.parquet"
            if shards == 1
            else f"{dataset_name}-{index:05d}-of-{shards:05d}.parquet"
        )
        partial = output_dir / f"{dataset_name}-{index:05d}.parquet.partial"
        partial.rename(output_dir / filename)
    return {"rows": count, "bytes": size, "shards": shards}
