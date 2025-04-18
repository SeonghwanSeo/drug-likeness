from argparse import ArgumentParser

from druglikeness import DrugLikeness


def parse_args():
    parser = ArgumentParser(description="Calculate Drug-likeness With Model")
    parser.add_argument("test_file", type=str, help="input smiles file (.smi)")
    parser.add_argument("-o", "--output", type=str, required=True, help="result file (.csv)")
    parser.add_argument("-m", "--model", type=str, help="model path", default="base")
    parser.add_argument("--naive", action="store_true", help="If True, model only considers one steroisomer")
    parser.add_argument("--cuda", action="store_true", help="If True, use cuda acceleration")
    parser.add_argument("--batch_size", type=int, help="Screening batch size", default=64)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = "cuda" if args.cuda else "cpu"
    model = DrugLikeness.from_pretrained(args.model, device)

    with open(args.test_file) as f:
        smiles_list = [ln.split()[0] for ln in f.readlines()]

    print(f"Screening {len(smiles_list)} SMILES")
    score_list = model.screening(smiles_list, args.naive, batch_size=args.batch_size, verbose=True)

    with open(args.output, "w") as w:
        w.write("SMILES,Score\n")
        for smi, score in zip(smiles_list, score_list, strict=True):
            w.write(f"{smi},{score:.3f}\n")
