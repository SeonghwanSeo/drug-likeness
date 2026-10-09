"""Convert whitespace-delimited SMILES files to a Hugging Face dataset."""

from collections.abc import Iterable, Iterator
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union

from datasets import Dataset, Features, List, Value

from ..model import canonicalize
from ..tokenizer import SmilesTokenizer


def _token_rows(
    paths: list[str],
    tokenizer: SmilesTokenizer,
    smiles_column: int,
    canonical: bool,
    max_length: int,
) -> Iterator[dict[str, list[int]]]:
    for path in paths:
        with open(path) as source:
            for line in source:
                smiles = line.split()[smiles_column]
                if canonical:
                    smiles = canonicalize(smiles)
                tokens = tokenizer.encode(smiles)
                if len(tokens) - 2 <= max_length:
                    yield {"token_ids": tokens}


def prepare_smiles(
    paths: Iterable[Union[str, Path]],
    output_dir: Union[str, Path],
    tokenizer: SmilesTokenizer,
    *,
    smiles_column: int = 0,
    canonical: bool = True,
    max_length: int = 127,
) -> dict[str, int]:
    """Save token IDs including BOS/EOS; omit molecules beyond the length limit."""
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir.parent) as cache:
        dataset = Dataset.from_generator(
            _token_rows,
            cache_dir=cache,
            features=Features({"token_ids": List(Value("uint8"))}),
            gen_kwargs={
                "paths": [str(path) for path in paths],
                "tokenizer": tokenizer,
                "smiles_column": smiles_column,
                "canonical": canonical,
                "max_length": max_length,
            },
        )
        dataset.save_to_disk(str(output_dir))
        return {"rows": len(dataset)}
