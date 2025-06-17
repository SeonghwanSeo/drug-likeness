from argparse import ArgumentParser
from pathlib import Path

from druglikeness.deepdl import DeepDL

FDA = "./data/test/fda.smi"
INVESTIGATION = "./data/test/investigation.smi"
CHEMBL = "./data/test/chembl.smi"
ZINC15 = "./data/test/zinc15.smi"
GDB17 = "./data/test/gdb17.smi"


def parse_args():
    parser = ArgumentParser(description="Calculate Drug-likeness With Model")
    parser.add_argument("-m", "--model", type=str, default="extended", help="Model name or path")
    parser.add_argument("-a", "--arch", type=str, default="deepdl", help="Architecture of the model.")
    parser.add_argument("--naive", action="store_true", help="If True, model only considers one steroisomer")
    parser.add_argument("--cuda", action="store_true", help="If True, use cuda acceleration")
    parser.add_argument("--batch_size", type=int, help="Screening batch size", default=64)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = "cuda" if args.cuda else "cpu"
    model = DeepDL.from_pretrained(args.model, device)

    for test_file in [FDA, INVESTIGATION, CHEMBL, ZINC15, GDB17]:
        name = Path(test_file).stem
        with open(test_file) as f:
            smiles_list = [ln.split()[0] for ln in f.readlines()]
        print(f"Test {len(smiles_list)} molecules in {test_file}")
        score_list = model.screening(smiles_list, args.naive, batch_size=args.batch_size, verbose=True)
        assert len(smiles_list) == len(score_list), "The number of SMILES and scores do not match."
        print("Average score", sum(score_list) / len(score_list))
        print()
