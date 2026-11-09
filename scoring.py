from argparse import ArgumentParser

from druglikeness.sdk.api import DrugLikenessClient


def parse_args():
    parser = ArgumentParser(description="Calculate Drug-likeness With Model")
    parser.add_argument("smiles", type=str, help="test smiles")
    parser.add_argument(
        "-a",
        "--arch",
        type=str,
        default="deepdl",
        choices=["deepdl", "chemprior"],
        help="Architecture of the model.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Model name or path (default: 'extended' for DeepDL, "
        "'large-chembl' for ChemPrior)",
    )
    parser.add_argument(
        "--mode",
        help="DeepDL: strict | naive; ChemPrior: original | first | best | remove",
    )
    args = parser.parse_args()
    modes = (
        ("original", "first", "best", "remove")
        if args.arch == "chemprior"
        else ("strict", "naive")
    )
    if args.mode is None:
        args.mode = modes[0]
    elif args.mode not in modes:
        parser.error(f"--mode for {args.arch} must be one of: {', '.join(modes)}")
    if args.model is None:
        args.model = "large-chembl" if args.arch == "chemprior" else "extended"
    return args


def construct_model(arch: str, model: str, device: str) -> DrugLikenessClient:
    if arch == "deepdl":
        from druglikeness.deepdl import DeepDL

        return DeepDL.from_pretrained(model, device)

    elif arch == "chemprior":
        from druglikeness.chemprior import ChemPrior

        return ChemPrior.from_pretrained(model, device)

    else:
        raise ValueError(f"Unknown architecture {arch}. Use 'deepdl' or 'chemprior'.")


if __name__ == "__main__":
    args = parse_args()
    model = construct_model(args.arch, args.model, "cpu")
    stereo = (
        {"stereo_mode": args.mode}
        if args.arch == "chemprior"
        else {"naive": args.mode == "naive"}
    )
    score = model.scoring(args.smiles, **stereo)
    print(f"score: {score:.3f}")
