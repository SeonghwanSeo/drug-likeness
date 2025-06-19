from argparse import ArgumentParser
from pathlib import Path

from druglikeness.sdk.api import DrugLikenessClient

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


def construct_model(args) -> DrugLikenessClient:
    device = "cuda" if args.cuda else "cpu"

    if args.arch == "deepdl":
        from druglikeness.deepdl import DeepDL

        return DeepDL.from_pretrained(args.model, device)

    else:
        raise ValueError(f"Unknown architecture {args.arch}. Supported are 'deepdl' and 'doubledeepdl'.")


def compute_auroc(true_scores: list[float], false_scores: list[float], high_is_better: bool = True) -> float:
    """Determines a ROC curve"""
    assert len(true_scores) > 0, "true_scores must not be empty"
    assert len(false_scores) > 0, "false_scores must not be empty"

    scores = true_scores + false_scores
    labels: list[bool] = [True] * len(true_scores) + [False] * len(false_scores)
    datas = sorted(zip(scores, labels), reverse=high_is_better)

    num_datas = len(datas)
    TP: list[int] = [0]  # True positive
    FP: list[int] = [0]  # False positive

    # loop over score list
    num_trues: int = 0
    num_falses: int = 0
    for i in range(num_datas):
        if datas[i][1]:
            num_trues += 1
        else:
            num_falses += 1
        TP.append(num_trues)  # TP
        FP.append(num_falses)  # FP
    assert num_trues == len(true_scores), "Number of true scores does not match the number of true labels"
    assert num_falses == len(false_scores), "Number of false scores does not match the number of false labels"

    # normalize, check that there are actives and inactives
    TPR = [i / num_trues for i in TP]  # True positive rate: TP/(TP+FN)
    FPR = [i / num_falses for i in FP]  # False positive rate: TP/(TN+FP)

    # loop over score list
    AUC: float = 0
    for i in range(0, num_datas - 1):
        AUC += (FPR[i + 1] - FPR[i]) * (TPR[i + 1] + TPR[i]) / 2
    return AUC


if __name__ == "__main__":
    args = parse_args()

    model = construct_model(args)

    results: dict[str, list[float]] = {}

    for test_file in [FDA, INVESTIGATION, CHEMBL, ZINC15, GDB17]:
        name = Path(test_file).stem
        with open(test_file) as f:
            smiles_list = [ln.split()[0] for ln in f.readlines()]
        print(f"Test {len(smiles_list)} molecules in {test_file}")
        score_list = model.screening(smiles_list, args.naive, batch_size=args.batch_size, verbose=True)
        assert len(smiles_list) == len(score_list), "The number of SMILES and scores do not match."
        print("Average score", sum(score_list) / len(score_list))
        print()
        results[name] = score_list

    print("fda vs chembl")
    print("AUROC", compute_auroc(results["fda"], results["chembl"]))
    print("fda vs zinc15")
    print("AUROC", compute_auroc(results["fda"], results["zinc15"]))
    print("fda vs gdb17")
    print("AUROC", compute_auroc(results["fda"], results["gdb17"]))
