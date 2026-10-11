import argparse
import json
from pathlib import Path

from druglikeness.chemprior.train_utils.preprocess import smiles_to_parquet


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert SMILES and source IDs to Zstd Parquet."
    )
    parser.add_argument(
        "inputs", nargs="+", type=Path, help="Tab-separated .smi or .smi.zst files."
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="New Parquet directory."
    )
    parser.add_argument("--max_shard_size_mb", type=int, default=500)
    parser.add_argument(
        "--dataset_name",
        default="train",
        help="Distribution filename prefix (default: train).",
    )
    parser.add_argument("--smiles_column", type=int, default=0)
    parser.add_argument(
        "--id_column", type=int, default=1, help="Source ID column (default: 1)."
    )
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
        id_column=args.id_column,
        dataset_name=args.dataset_name,
        canonical=args.canonical,
        compression_level=args.compression_level,
        max_shard_size=args.max_shard_size_mb * 1_000_000,
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
