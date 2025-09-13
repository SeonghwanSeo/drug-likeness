from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import gdown
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
from typing_extensions import Self

from druglikeness.sdk.api import DrugLikenessClient, ModelConfig

from .vocab import EOS_TOKEN, EOS_TOKEN_ID, PAD_TOKEN, PAD_TOKEN_ID, VOCAB

NAMES = {
    # model trained on updated drug dataset excluding fda-approved
    "extended": "extended-20250618",
    # model trained on worlddrug-not-fda used in paper (2833 molecules)
    "chemsci-2025": "chemsci-20250618",
    # model (paper; worlddrug finetuned)
    "chemsci-2021": "chemsci-2021-pubchem_worlddrug",
    # pretrained model on pubchem dataset (paper)
    "chemsci-2021-pretrain": "chemsci-2021-pubchem",
}

GOOGLE_DRIVE_IDS = {
    "extended-20250618": "1Qz-XOHHwE1tBhh_0MWASRM6B6vpXAUxX",
    "chemsci-20250618": "1ggTXcLD_AhRGYkoByKWqMG8-080BDq6_",
    "chemsci-2021-pubchem_worlddrug": "1TMJ9ZjI1x7yey-x-qdT8Veg_sUVRgdcH",
    "chemsci-2021-pubchem": "1GI3fO0ndF-2yN5pMNDAHFBXyBa8xXtmc",
}


@dataclass
class DeepDLConfig(ModelConfig):
    input_size: int = 67
    hidden_size: int = 1024
    n_layers: int = 4
    dropout: float = 0.2


class DeepDL(DrugLikenessClient[DeepDLConfig]):
    def __init__(self, config: Optional[DeepDLConfig] = None):
        super().__init__()
        if config is None:
            config = DeepDLConfig()
        self.input_size: int = config.input_size
        self.hidden_size: int = config.hidden_size
        self.n_layers: int = config.n_layers
        self.dropout: float = config.dropout

        self.embedding: nn.Embedding = nn.Embedding(self.input_size, self.hidden_size)
        self.start_codon: nn.Parameter = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.GRU = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            dropout=self.dropout,
            batch_first=True,
        )
        self.fc: nn.Linear = nn.Linear(self.hidden_size, self.input_size)
        self.vocab: dict[str, int] = VOCAB
        self.eos_token: str = EOS_TOKEN
        self.pad_token: str = PAD_TOKEN
        self.eos_token_id: int = EOS_TOKEN_ID
        self.pad_token_id: int = PAD_TOKEN_ID

    @property
    def device(self) -> torch.device:
        return self.start_codon.device

    # ==== Interface ==== #
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str = "extended",
        device: Union[torch.device, str] = "cpu",
        config: Optional[DeepDLConfig] = None,
    ) -> Self:
        # set model params

        device = torch.device(device)
        if pretrained_model_name_or_path in NAMES:
            model_name = NAMES[pretrained_model_name_or_path]
            checkpoint_path = Path.home() / ".local/share/deepdl" / (model_name + ".ckpt")
            if not checkpoint_path.exists():
                id = GOOGLE_DRIVE_IDS[model_name]
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                gdown.download(id=id, output=str(checkpoint_path), quiet=False)
        else:
            checkpoint_path = Path(pretrained_model_name_or_path)
            assert checkpoint_path.exists(), (
                f"Model path {checkpoint_path} does not exist., Supported Models: {list(NAMES.keys())}"
            )

        model = cls(config)
        model = model.to(device)

        state_dict = torch.load(checkpoint_path, map_location=device.type, weights_only=False)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict)
        model.eval()
        return model

    @torch.no_grad()
    def screening(
        self,
        smiles_list: list[str],
        naive: bool = False,
        batch_size: int = 64,
        verbose: bool = False,
    ) -> list[float]:
        self.GRU.flatten_parameters()  # for performance

        # sorting for efficient batching
        sorted_smiles_list = sorted(smiles_list, key=lambda x: (len(x), x), reverse=True)

        indices: list[tuple[int, int]] = []
        all_smiles: list[str] = []
        ofs: int = 0
        for smi in sorted_smiles_list:
            # filter out invalid SMILES
            if not (set(smi) <= self.vocab.keys()):
                indices.append((ofs, 0))
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                indices.append((ofs, 0))
                continue

            # scoring
            if naive:
                isomers = [Chem.MolToSmiles(next(EnumerateStereoisomers(mol)))]
            else:
                isomers = [Chem.MolToSmiles(isomer) for isomer in EnumerateStereoisomers(mol)]
            all_smiles.extend(isomers)
            indices.append((ofs, len(isomers)))
            ofs += len(isomers)

        iterator = tqdm(range(0, len(all_smiles), batch_size), desc="screening", unit="batch", disable=not verbose)
        flatten_scores: list[float] = sum(
            [self.evaluate_batch(all_smiles[i : i + batch_size]) for i in iterator], start=[]
        )
        smi_to_scores: dict[str, float] = {}
        assert len(sorted_smiles_list) == len(indices), "Mismatch in smiles and indices length."
        for smi, (ofs, num_isomers) in zip(sorted_smiles_list, indices):
            if num_isomers == 0:
                # for invalid SMILES
                score = 0
            else:
                # set lowest score among isomers
                score = min(flatten_scores[ofs : ofs + num_isomers])
            smi_to_scores[smi] = score
        return [smi_to_scores[smi] for smi in smiles_list]

    # ===== Internal functions ===== #

    def forward(self, seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq: [B, L] where B is the batch size and L is the length of SMILES.
            lengths: [B,] where B is the batch size.
        Returns:
            logits: [B, L, C] where C is the number of characters.
        """
        # prepare input; shift seq by one to the right
        # input: [N, C, =, O, <EOS>]
        # shift: [<BOS>, N, C, =, O]
        seq = seq[:, :-1]  # [B, L-1]; remove the last token (eos)
        x = self.embedding(seq)  # [B, L-1] => [B, L-1, F]
        start_codon = self.start_codon.expand_as(x[:, :1, :])  # [B, 1, F]
        x = torch.cat([start_codon, x], 1)  # [B, L, F]

        packed_seq = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_x, _ = self.GRU(packed_seq)

        packed_logits = PackedSequence(self.fc(packed_x.data), packed_seq.batch_sizes)  # [Lpacked, F] => [Lpacked, C]
        logits, _ = pad_packed_sequence(packed_logits, batch_first=True)  # [B, L, F]
        return logits

    def calc_p_x(self, seq: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq: [B, L] where B is the batch size and L is the length of SMILES.
            logits: [B, L, C] where C is the number of characters.
        Returns:
            log_p_seq: [B,]
        """
        logits[:, :, self.pad_token_id] = -torch.inf  # mask padding
        log_p_chars = logits.log_softmax(dim=-1)  # [B, L, C]

        log_p_tokens = torch.gather(log_p_chars, dim=2, index=seq.unsqueeze(-1)).squeeze(-1)  # [B, L]
        log_p_tokens[log_p_tokens.isneginf()] = 0.0  # mask padding
        log_p_seq = log_p_tokens.sum(dim=-1)  # [B,]
        return log_p_seq

    def encode(
        self,
        input: list[str],
        max_length: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encodes SMILES strings into a tensor of token IDs.
        Args:
            input: A list of SMILES strings.
            max_length: Maximum length of the output tensor.

        Returns:
            tokenized_tensor: A tensor of shape [max_length, batch_size] containing token IDs.
            lengths: A length of each sequence
        """
        batch_size = len(input)
        assert batch_size > 0, "There is no input SMILES. Please provide at least one SMILES string."

        # add eos token to each sequence
        input = [seq + self.eos_token for seq in input]

        length_list = [len(seq) for seq in input]
        if max_length is None:
            max_length = max(length_list)

        tokenized: list[list[int]] = []
        for smi, length in zip(input, length_list):
            tk_smi = [self.vocab[c] for c in smi] + [self.pad_token_id] * (max_length - length)
            tokenized.append(tk_smi)
        tokenized_tensor = torch.tensor(tokenized, dtype=torch.long, device=self.device)  # [B, L]
        lengths = torch.tensor(length_list, dtype=torch.long)  # [B,]
        return tokenized_tensor, lengths

    def evaluate_batch(self, smiles_list: list[str]) -> list[float]:
        # sort SMILES by length in descending order
        indices = sorted(range(len(smiles_list)), key=lambda i: len(smiles_list[i]), reverse=True)
        smiles_list = [smiles_list[i] for i in indices]

        x, lengths = self.encode(smiles_list)
        score_list = self._calc_scores_batch(x, lengths).tolist()

        # remap SMILES to original order
        indice_score_list = sorted(list(zip(indices, score_list)))
        score_list = [score for _, score in indice_score_list]
        return score_list

    def _calc_scores_batch(self, seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Calculate normalized scores for a batch of SMILES.

        Parameters
        ----------
        seq : torch.Tensor
            tensor of SMILES token IDs.
            shape: [max length, batch size]
        lengths: torch.Tensor
            Length of each sequence.
            shape: [batch size]

        Returns
        -------
        list[float]
            scores for each SMILES in the batch.

        """
        logits = self.forward(seq, lengths)
        log_p_seq = self.calc_p_x(seq, logits)  # [L, B, C]
        # normalize scores \in [0, 100]
        return torch.clamp(log_p_seq + 100, min=0.0)  # [B,]
