import random
from pathlib import Path
from typing import Union

import torch
from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from typing_extensions import Self

from druglikeness.deepdl.vocab import EOS_TOKEN, PAD_TOKEN_ID, VOCAB


class SmilesDataset(Dataset):
    def __init__(self, smiles_list: list[str], preprocess: bool = True):
        self.vocab: dict[str, int] = VOCAB
        self.eos_token: str = EOS_TOKEN

        if preprocess:
            smiles_list = self.preprocess(smiles_list)
        self.smiles_list: list[str] = smiles_list

    @classmethod
    def from_file(cls, smi_path: Union[str, Path], preprocess: bool = True) -> Self:
        with open(smi_path) as f:
            smiles_list: list[str] = [ln.strip().split()[0] for ln in f.readlines()]
        # preprocess the SMILES strings
        smiles_list = cls.preprocess(smiles_list) if preprocess else smiles_list
        return cls(smiles_list, preprocess=False)

    @staticmethod
    def preprocess(smiles_list: list[str]) -> list[str]:
        refine_smiles_list: list[str] = []
        for smi in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smi)
                smi = Chem.MolToSmiles(mol)
                assert mol is not None and smi is not None and len(smi) > 0
            except Exception as e:
                print(e)
                continue
            refine_smiles_list.append(smi)
        return refine_smiles_list

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx: int) -> torch.Tensor:
        smi = self.smiles_list[idx]
        # random stereo-isomer
        isomers = list(EnumerateStereoisomers(Chem.MolFromSmiles(smi)))
        smi = Chem.MolToSmiles(isomers[random.randint(0, len(isomers) - 1)], isomericSmiles=True)
        smi += self.eos_token
        sequence = torch.tensor([self.vocab[c] for c in smi])
        return sequence


def collate_fn(batch: list[torch.Tensor], pad_token_id=PAD_TOKEN_ID) -> tuple[torch.Tensor, torch.Tensor]:
    # sort by length in descending order (RNN expects sequences in descending order)
    batch = sorted(batch, key=lambda x: x.size(0), reverse=True)
    seqs = pad_sequence(batch, batch_first=True, padding_value=pad_token_id)
    lengths = torch.tensor([seq.size(0) for seq in batch], dtype=torch.long)
    return seqs, lengths
