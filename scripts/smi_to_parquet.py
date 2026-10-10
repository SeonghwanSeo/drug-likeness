import argparse
import json
from pathlib import Path

from druglikeness.deepdl2.train_utils.preprocess import smiles_to_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SMILES files to Zstd Parquet.")
    parser.add_argument("inputs", nargs="+", type=Path, help=".smi or .smi.zst files.")
    parser.add_argument(
        "--output", required=True, type=Path, help="New Parquet directory."
    )
    parser.add_argument("--max_shard_size_mb", type=int, default=500)
    parser.add_argument("--smiles_column", type=int, default=0)
    parser.add_argument("--compression_level", type=int, default=5)
    parser.add_argument(
        "--no_canonical",
        dest="canonical",
        action="store_false",
        help="Input SMILES are already canonicalized.",
    )
    args = parser.parse_args()
    report = smiles_to_parquet(
        args.inputs,
        args.output,
        smiles_column=args.smiles_column,
        canonical=args.canonical,
        compression_level=args.compression_level,
        max_shard_size=args.max_shard_size_mb * 1_000_000,
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
