import argparse
import json
from pathlib import Path

from druglikeness.deepdl2.train_utils.preprocess import parquet_to_arrow


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream SMILES Parquet to HF Arrow without a temporary Arrow copy."
    )
    parser.add_argument("input", type=Path, help="Input Parquet directory.")
    parser.add_argument(
        "--output", required=True, type=Path, help="New HF dataset directory."
    )
    parser.add_argument("--num_shards", type=int, help="Number of final Arrow files.")
    parser.add_argument("--num_proc", type=int, help="Worker processes (default: 1).")
    args = parser.parse_args()
    report = parquet_to_arrow(
        args.input, args.output, num_shards=args.num_shards, num_proc=args.num_proc
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
