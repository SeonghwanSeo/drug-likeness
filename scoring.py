from argparse import ArgumentParser

from druglikeness.sdk.api import DrugLikenessClient


def parse_args():
    parser = ArgumentParser(description="Calculate Drug-likeness With Model")
    parser.add_argument("smiles", type=str, help="test smiles")
    parser.add_argument("-m", "--model", type=str, default="extended", help="Model name or path")
    parser.add_argument("-a", "--arch", type=str, default="deepdl", help="Architecture of the model.")
    parser.add_argument("--naive", action="store_true", help="If True, model only considers one steroisomer")
    args = parser.parse_args()
    return args


def construct_model(arch: str, model: str, device: str) -> DrugLikenessClient:
    if arch == "deepdl":
        from druglikeness.deepdl import DeepDL

        return DeepDL.from_pretrained(model, device)

    else:
        raise ValueError(f"Unknown architecture {arch}. Supported is 'deepdl'.")


if __name__ == "__main__":
    args = parse_args()
    model = construct_model(args.arch, args.model, "cpu")
    score = model.scoring(args.smiles, naive=args.naive)
    print(f"score: {score:.3f}")
