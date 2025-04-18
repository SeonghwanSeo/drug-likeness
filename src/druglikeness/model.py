from pathlib import Path

import gdown
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers

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
        dropout = params["dropout"]
        self.stereo = params["stereo"]

        self.GRU = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=n_layers, dropout=dropout)
        self.GRU.flatten_parameters()
        self.embedding: nn.Embedding = nn.Embedding(input_size, hidden_size)
        self.start_codon: nn.Parameter = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.fc: nn.Linear = nn.Linear(hidden_size, input_size)
        self.eval()

    @property
    def device(self) -> torch.device:
        return self.start_codon.device

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, device: str | torch.device = "cpu"):
        # since this is the wrapping calss of previous repository,
        # I need to keep this complicated structure...
        if pretrained_model_name_or_path in API_KEY:
            model_name = pretrained_model_name_or_path
            checkpoint_dir = Path(__file__).parent.parent.parent / "weights" / model_name
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
        try:
            model.load_state_dict(torch.load(weight_path, map_location="cpu"))
        except Exception:
            model.start_codon = nn.Parameter(model.start_codon.view(-1))
            model.load_state_dict(torch.load(weight_path, map_location="cpu"))
            model.start_codon = nn.Parameter(model.start_codon.view(1, 1, -1))
        model = model.to(device)
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encoding
        x = self.embedding(x)  # [N, L] => [N, L, F]
        x = x.permute(1, 0, 2)  # [N, L, F] => [L, N, F]
        start_codon = self.start_codon.repeat(1, x.size(1), 1)  # [1, N, F]
        x = torch.cat([start_codon, x], 0)  # [L+1, N, F]
        retval, _ = self.GRU(x)
        retval = retval.permute(1, 0, 2)  # [L+1, N, F] => [N, L+1, F]

        # decoding
        retval = self.fc(retval)  # [N, L+1, F] => [N, L+1, C]
        return retval

    @staticmethod
    def len_mask(lengths, max_length):
        """
        Mask the padding part of the result

        example data:
        c1ccccc1Q_________
        c1ccccc1Cc2ccccc2Q
        CNQ_______________
        ...

        We set the value of Padding part to 0
        """
        device = lengths.device
        mask = torch.arange(0, max_length, device=device).repeat(lengths.size(0))
        lengths = lengths.unsqueeze(1).repeat(1, max_length).reshape(-1)
        mask = mask - lengths
        mask[mask >= 0] = 0
        mask[mask < 0] = 1
        return mask

    @torch.no_grad()
    def evaluate(self, smiles: str, naive: bool = False) -> float:
        try:
            score = self._evaluate(smiles, naive)
        except KeyboardInterrupt as e:
            raise e
        except KeyError:
            score = 0
        except AssertionError:
            score = 0
        return score

    @torch.no_grad()
    def _evaluate(self, smiles: str, naive: bool = False) -> float:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        if self.stereo:
            if naive:
                isomers = [Chem.MolToSmiles(next(EnumerateStereoisomers(mol)))]
            else:
                isomers = [Chem.MolToSmiles(isomer) for isomer in EnumerateStereoisomers(mol)]
        else:
            isomers = [Chem.MolToSmiles(mol)]

        score = max(self._scoring(smi) for smi in isomers)
        return max(0, score + 100)  # normalize: 0 to 100

    def _scoring(self, smi: str) -> float:
        smi = smi + "Q"  # add eos token
        token_ids = [C_TO_I[i] for i in smi]
        x = torch.tensor(token_ids, device=self.device)
        output = self.forward(x.unsqueeze(0)).squeeze(0)
        p_char = output.log_softmax(dim=-1)
        p_char = p_char.cpu().numpy()
        score = sum(p_char[i, token_id] for i, token_id in enumerate(token_ids))
        return float(score)
