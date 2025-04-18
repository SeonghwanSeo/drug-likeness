from pathlib import Path

import gdown
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
from tqdm import tqdm

from druglikeness.data_utils import C_TO_I, N_CHAR

API_KEY = {
    "base": "1EyxROAOGic40Q0MT2LM7bpJxy_bJttQQ",
    "pubchem_worlddrug": "1EyxROAOGic40Q0MT2LM7bpJxy_bJttQQ",
    "worlddrug": "1dtfUmamh3MAiATKXGoLyDsMEYHfLxHN0",
}


class DrugLikeness(nn.Module):
    # Static attribute
    default_parameters = {"input_size": N_CHAR, "stereo": True, "hidden_size": 1024, "n_layers": 4, "dropout": 0.2}

    def __init__(self, **params):
        super().__init__()
        if params is None:
            params = self.default_parameters
        else:
            params = self.default_parameters | params
        self.params = params

        input_size = params["input_size"]
        hidden_size = params["hidden_size"]
        n_layers = params["n_layers"]
        self.stereo = params["stereo"]

        self.GRU = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=n_layers)
        self.embedding: nn.Embedding = nn.Embedding(input_size, hidden_size)
        self.start_codon: nn.Parameter = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.fc: nn.Linear = nn.Linear(hidden_size, input_size)
        self.eval()

        self.c_to_i: dict[str, int] = C_TO_I

        self.__cache_start_codon: torch.Tensor | None = None
        self.is_flatten: bool = False

    @property
    def device(self) -> torch.device:
        return self.start_codon.device

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, device: str | torch.device = "cpu"):
        # since this is the wrapping calss of previous repository,
        # I need to keep this complicated structure...
        if pretrained_model_name_or_path in API_KEY:
            model_name = pretrained_model_name_or_path
            checkpoint_dir = Path(__file__).parent / "weights" / model_name
            if not checkpoint_dir.exists():
                id = API_KEY[model_name]
                gdown.download_folder(id=id, output=str(checkpoint_dir), quiet=False)
        else:
            checkpoint_dir = Path(pretrained_model_name_or_path)
            assert checkpoint_dir.exists(), f"Model path {checkpoint_dir} does not exist."

        weight_path = checkpoint_dir / "save.pt"
        config_path = checkpoint_dir / "config.yaml"
        model_param = OmegaConf.load(config_path).model
        model = cls(**model_param)
        # TODO: save the model file with modified structure
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
        try:
            model.load_state_dict(state_dict)
        except Exception:
            model.start_codon = nn.Parameter(model.start_codon.view(-1))
            model.load_state_dict(state_dict)
            model.start_codon = nn.Parameter(model.start_codon.view(1, 1, -1))
        model = model.to(device)
        return model

    def tokenize(self, smi: str, max_length: int | None = None) -> list[int]:
        if max_length is None:
            smi = smi + "Q"  # add eos token
        else:
            assert max_length >= (len(smi) + 1)
            smi = smi + "Q" * (max_length - len(smi))
        return [self.c_to_i[c] for c in smi]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_flatten:
            self.GRU.flatten_parameters()
            self.is_flatten = True

        N = x.shape[0]
        if self.__cache_start_codon is None:
            self.__cache_start_codon = self.start_codon.data.repeat(1, N, 1)
        else:
            if self.__cache_start_codon.shape[1] < N:
                self.__cache_start_codon = self.start_codon.data.repeat(1, N, 1)
        start_codon = self.__cache_start_codon[:, :N, :]

        # encoding
        x = x.permute(1, 0)  # [N, L] => [L, N]
        x = self.embedding(x)  # [L, N] => [L, N, F]
        x = torch.cat([start_codon, x], 0)  # [L+1, N, F]
        retval, _ = self.GRU(x)

        # decoding
        logits = self.fc(retval)  # [L+1, N, F] => [L+1, N, C]
        logits = logits.permute(1, 0, 2)  # [L+1, N, C] => [N, L+1, C]
        return logits

    def calc_score(self, token_ids: list[int], logp_char: np.ndarray) -> float:
        logp = sum(logp_char[i, token_id] for i, token_id in enumerate(token_ids))
        score = max(0, float(logp) + 100)  # normalize: 0 to 100
        return score

    @torch.no_grad()
    def screening(
        self,
        smiles_list: list[str],
        naive: bool = False,
        batch_size: int = 64,
        verbose: bool = False,
    ) -> list[float]:
        # sorting for efficient batching
        sorted_smiles_list = sorted(smiles_list, key=lambda x: (len(x), x))

        indices = []
        all_smiles = []
        ofs = 0
        for smi in sorted_smiles_list:
            if any(c not in self.c_to_i for c in smi):
                indices.append((ofs, 0))
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                indices.append((ofs, 0))
                continue
            else:
                if self.stereo:
                    if naive:
                        isomers = [Chem.MolToSmiles(next(EnumerateStereoisomers(mol)))]
                    else:
                        isomers = [Chem.MolToSmiles(isomer) for isomer in EnumerateStereoisomers(mol)]
                else:
                    isomers = [Chem.MolToSmiles(mol)]
                all_smiles.extend(isomers)
                indices.append((ofs, len(isomers)))
                ofs += len(isomers)

        _iter = tqdm(range(0, len(all_smiles), batch_size), desc="screening", unit="batch", disable=not verbose)
        flatten_scores: list[float] = sum(
            [self._batch_evaluate(all_smiles[i : i + batch_size]) for i in _iter], start=[]
        )
        smi_to_scores: dict[str, float] = {}
        for smi, (ofs, num_isomers) in zip(sorted_smiles_list, indices, strict=True):
            if num_isomers == 0:
                score = 0
            else:
                score = max(flatten_scores[ofs : ofs + num_isomers])
            smi_to_scores[smi] = score
        return [smi_to_scores[smi] for smi in smiles_list]

    @torch.no_grad()
    def scoring(self, smiles: str, naive: bool = False) -> float:
        try:
            score = self._scoring(smiles, naive)
        except KeyboardInterrupt as e:
            raise e
        except KeyError:
            score = 0
        except AssertionError:
            score = 0
        return score

    @torch.no_grad()
    def _scoring(self, smiles: str, naive: bool = False) -> float:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0
        if self.stereo:
            if not naive:
                isomers = [Chem.MolToSmiles(isomer) for isomer in EnumerateStereoisomers(mol)]
                iters = (isomers[i : i + 16] for i in range(0, len(isomers), 16))
                return max(max(self._batch_evaluate(chunk)) for chunk in iters)
            else:
                smi = Chem.MolToSmiles(next(EnumerateStereoisomers(mol)))
                return self._evaluate(smi)
        else:
            smi = Chem.MolToSmiles(mol)
            return self._evaluate(smi)

    def _evaluate(self, smi: str) -> float:
        token_ids = self.tokenize(smi)
        x = torch.tensor(token_ids, device=self.device)
        logits = self.forward(x.unsqueeze(0)).squeeze(0)
        logp_char = logits.log_softmax(dim=-1).cpu().numpy()
        return self.calc_score(token_ids, logp_char)

    def _batch_evaluate(self, smiles_list: list[str]) -> list[float]:
        max_length = max(len(smi) for smi in smiles_list) + 1
        token_ids_list = [self.tokenize(smi, max_length) for smi in smiles_list]
        x = torch.tensor(token_ids_list, device=self.device)
        logits = self.forward(x)
        logp_chars = logits.log_softmax(dim=-1).cpu().numpy()

        # TODO: is there any way better than iteration?
        scores: list[float] = []
        for idx, smi in enumerate(smiles_list):
            token_ids = token_ids_list[idx][: len(smi) + 1]
            logp_char = logp_chars[idx, : len(smi) + 1]
            scores.append(self.calc_score(token_ids, logp_char))
        return scores
