from argparse import ArgumentParser

from druglikeness import DrugLikeness


def parse_args() -> tuple[str, str, bool]:
    parser = ArgumentParser(description="Calculate Drug-likeness With Model")
    parser.add_argument("smiles", type=str, help="test smiles")
    parser.add_argument("-m", "--model", type=str, help="model path", default="base")
    parser.add_argument("--naive", action="store_true", help="If True, model only considers one steroisomer")
    args = parser.parse_args()
    return args.smiles, args.model, args.naive


if __name__ == "__main__":
    smiles, model_path, naive = parse_args()
    model = DrugLikeness.from_pretrained(model_path, "cpu")
    score = model.evaluate(smiles, naive=naive)
    print(f"score: {score:.3f}")
