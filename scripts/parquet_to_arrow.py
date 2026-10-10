import argparse
import json
from pathlib import Path

from druglikeness.deepdl2.train_utils.preprocess import parquet_to_arrow


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SMILES Parquet to HF Arrow.")
    parser.add_argument("input", type=Path, help="Input Parquet directory.")
    parser.add_argument(
        "--output", required=True, type=Path, help="HF dataset directory."
    )
    parser.add_argument("--num_shards", type=int)
    parser.add_argument("--num_proc", type=int)
    args = parser.parse_args()
    report = parquet_to_arrow(
        args.input, args.output, num_shards=args.num_shards, num_proc=args.num_proc
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
