"""Prepare tokenized Arrow shards and their batch index."""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Union

import numpy as np

from ..tokenizer import SmilesTokenizer


def build_arrow_index(
    paths: Iterable[Union[str, Path]], output: Union[str, Path]
) -> dict[str, Any]:
    """Record Arrow batch locations and sequence lengths."""
    import pyarrow as pa
    from pyarrow import ipc

    index: dict[str, Any] = {"files": [], "blocks": [], "rows": 0, "max_length": 0}
    for file_number, path in enumerate(sorted(map(Path, paths))):
        index["files"].append({"path": str(path.resolve())})
        with pa.memory_map(str(path), "r") as source:
            reader = ipc.open_file(source)
            for batch_number in range(reader.num_record_batches):
                batch = reader.get_batch(batch_number)
                ids = batch.column(batch.schema.get_field_index("token_ids"))
                lengths = np.diff(ids.offsets.to_numpy())
                index["blocks"].append(
                    {"file": file_number, "batch": batch_number, "rows": len(lengths)}
                )
                index["rows"] += len(lengths)
                index["max_length"] = max(index["max_length"], int(lengths.max()) - 1)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(index, indent=2) + "\n")
    return index


def prepare_smiles(
    paths: Iterable[Union[str, Path]],
    output_dir: Union[str, Path],
    tokenizer: SmilesTokenizer,
    *,
    smiles_column: int = 0,
    batch_rows: int = 10000,
    shard_rows: int = 1000000,
    canonical: bool = True,
) -> dict[str, Any]:
    """Canonicalize/tokenize text in bounded chunks; write the ZINC Arrow schema.

    Input is whitespace-delimited with an explicit SMILES column; source IDs
    are not needed by the LM.
    Set canonical=False for the existing canonical corpora.
    """
    import pyarrow as pa
    from pyarrow import ipc

    from ..model import canonicalize

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [("token_ids", pa.list_(pa.uint8() if len(tokenizer) <= 256 else pa.uint16()))]
    )
    paths = [Path(path).resolve() for path in paths]
    files, buffer, writer, rows, shard_count = [], [], None, 0, 0
    try:
        for path in paths:
            with path.open() as source:
                for line in source:
                    smiles = line.split()[smiles_column]
                    if canonical:
                        smiles = canonicalize(smiles)
                    tokens = tokenizer.encode(smiles)
                    buffer.append(tokens)
                    if writer is None:
                        destination = output_dir / f"shard-{len(files):05d}.arrow"
                        files.append(destination)
                        writer = ipc.new_file(
                            str(destination),
                            schema,
                            options=ipc.IpcWriteOptions(compression="zstd"),
                        )
                    shard_count += 1
                    rows += 1
                    if len(buffer) >= batch_rows or shard_count >= shard_rows:
                        writer.write_batch(
                            pa.RecordBatch.from_arrays(
                                [pa.array(buffer, type=schema[0].type)], schema=schema
                            )
                        )
                        buffer.clear()
                    if shard_count >= shard_rows:
                        writer.close()
                        writer, shard_count = None, 0
        if buffer:
            writer.write_batch(
                pa.RecordBatch.from_arrays(
                    [pa.array(buffer, type=schema[0].type)], schema=schema
                )
            )
    finally:
        if writer is not None:
            writer.close()
    (output_dir / "summary.json").write_text(
        json.dumps({"rows": rows}, indent=2) + "\n"
    )
    tokenizer.save(output_dir / "vocab.txt")
    result = build_arrow_index(files, output_dir / "index.json")
    (output_dir / ".done").touch()
    return result
