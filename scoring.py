from argparse import ArgumentParser

from druglikeness import DrugLikeness


def parse_args():
    parser = ArgumentParser(description="Calculate Drug-likeness With Model")
    parser.add_argument("smiles", type=str, help="test smiles")
    parser.add_argument("-m", "--model", type=str, help="model path", default="base")
    parser.add_argument("--naive", action="store_true", help="If True, model only considers one steroisomer")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    model = DrugLikeness.from_pretrained(args.model, "cpu")
    score = model.scoring(args.smiles, naive=args.naive)
    print(f"score: {score:.3f}")
