from argparse import ArgumentParser

from druglikeness.sdk.api import DrugLikenessClient


def parse_args():
    parser = ArgumentParser(description="Calculate Drug-likeness With Model")
    parser.add_argument("test_file", type=str, help="input smiles file (.smi)")
    parser.add_argument("-o", "--output", type=str, required=True, help="result file (.csv)")
    parser.add_argument("-m", "--model", type=str, default="extended", help="Model name or path")
    parser.add_argument("-a", "--arch", type=str, default="deepdl", help="Architecture of the model.")
    parser.add_argument("--naive", action="store_true", help="If True, model only considers one steroisomer")
    parser.add_argument("--cuda", action="store_true", help="If True, use cuda acceleration")
    parser.add_argument("--batch_size", type=int, help="Screening batch size", default=64)
    return parser.parse_args()


def construct_model(arch: str, model: str, device: str) -> DrugLikenessClient:
    if arch == "deepdl":
        from druglikeness.deepdl import DeepDL

        return DeepDL.from_pretrained(model, device)

    else:
        raise ValueError(f"Unknown architecture {arch}. Supported is 'deepdl'.")


if __name__ == "__main__":
    args = parse_args()

    device = "cuda" if args.cuda else "cpu"
    model = construct_model(args.arch, args.model, device)

    with open(args.test_file) as f:
        smiles_list = [ln.split()[0] for ln in f.readlines()]

    print(f"Screening {len(smiles_list)} SMILES")
    score_list = model.screening(smiles_list, args.naive, batch_size=args.batch_size, verbose=True)
    assert len(smiles_list) == len(score_list), "The number of SMILES and scores do not match."

    with open(args.output, "w") as w:
        w.write("SMILES,Score\n")
        for smi, score in zip(smiles_list, score_list):
            w.write(f"{smi},{score:.3f}\n")
