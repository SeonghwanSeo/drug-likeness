"""Autoregressive SMILES models with the legacy DeepDL scoring interface.

forward receives right-padded targets (SMILES + EOS, no BOS) and shifts them
internally. Output t cannot observe target t or any later token.
"""

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import torch
from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from tqdm import tqdm
from typing_extensions import Self

from druglikeness.sdk.api import DrugLikenessClient, ModelConfig

from .tokenizer import VOCAB_PATH, SmilesTokenizer
from .transformer import TransformerStack


@dataclass
class DeepDL2Config(ModelConfig):
    architecture: str = "transformer"
    input_size: int = field(default_factory=lambda: len(SmilesTokenizer()))
    hidden_size: int = 384
    n_layers: int = 8
    n_heads: int = 6
    dropout: float = 0.1  # GRU only; the Transformer has no dropout

    @classmethod
    def preset(cls, name: str = "medium", **overrides: Any) -> Self:
        presets = {
            "small": dict(hidden_size=256, n_layers=8, n_heads=4),
            "medium": dict(hidden_size=384, n_layers=8, n_heads=6),
            "large": dict(hidden_size=512, n_layers=12, n_heads=8),
            "gru-medium": dict(
                architecture="gru", hidden_size=768, n_layers=4, dropout=0.2
            ),
        }
        return cls(**{**presets[name], **overrides})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonicalize(smiles: str) -> str:
    return Chem.MolToSmiles(
        Chem.MolFromSmiles(smiles), canonical=True, isomericSmiles=True
    )


class DeepDL2_Base(DrugLikenessClient[DeepDL2Config]):
    def __init__(
        self,
        config: Optional[DeepDL2Config] = None,
        tokenizer: Optional[SmilesTokenizer] = None,
    ) -> None:
        super().__init__()
        self.config: DeepDL2Config = config or DeepDL2Config()
        self.tokenizer: SmilesTokenizer = tokenizer or SmilesTokenizer()
        self.vocab: dict[str, int] = self.tokenizer.token_to_id
        self.pad_token_id: int = self.tokenizer.pad_token_id
        self.bos_token_id: int = self.tokenizer.bos_token_id
        self.eos_token_id: int = self.tokenizer.eos_token_id
        c = self.config
        self.embedding: nn.Embedding = nn.Embedding(c.input_size, c.hidden_size)

    @property
    def device(self) -> torch.device:
        return self.embedding.weight.device

    @classmethod
    def from_config(
        cls, config: DeepDL2Config, tokenizer: Optional[SmilesTokenizer] = None
    ) -> "DeepDL2_Base":
        return {"transformer": DeepDL2_TF, "gru": DeepDL2_GRU}[config.architecture](
            config, tokenizer
        )

    def prediction_logits(self, logits: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros(logits.size(-1), dtype=torch.bool, device=logits.device)
        mask[self.pad_token_id] = mask[self.bos_token_id] = True
        return logits.float().masked_fill(mask, -torch.inf)

    def calc_p_x(self, seq: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        values = (
            self.prediction_logits(logits)
            .log_softmax(-1)
            .gather(-1, seq.unsqueeze(-1))
            .squeeze(-1)
        )
        return values.masked_fill(seq.eq(self.pad_token_id), 0.0).sum(-1)

    def tokenize(self, smiles: str, add_special_tokens: bool = True) -> list[int]:
        """Return targets: lexical tokens plus EOS, without BOS (shifted internally)."""
        ids = self.tokenizer.encode(smiles, add_special_tokens=False)
        return [*ids, self.eos_token_id] if add_special_tokens else ids

    def encode(self, input: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        rows = [torch.tensor(self.tokenize(s), dtype=torch.long) for s in input]
        lengths = torch.tensor([len(row) for row in rows], dtype=torch.long)
        seq = pad_sequence(rows, batch_first=True, padding_value=self.pad_token_id)
        return seq.to(self.device), lengths

    @torch.no_grad()
    def log_likelihood(
        self, smiles_list: list[str], batch_size: int = 64, canonical: bool = True
    ) -> list[float]:
        """Unclipped summed log p(SMILES, EOS); invalid/unsupported input raises."""
        strings = (
            [canonicalize(s) for s in smiles_list] if canonical else list(smiles_list)
        )
        order = sorted(range(len(strings)), key=lambda i: len(strings[i]))
        scores = [0.0] * len(strings)
        training = self.training
        self.eval()
        try:
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                seq, lengths = self.encode([strings[i] for i in indices])
                values = self.calc_p_x(seq, self(seq, lengths)).tolist()
                for i, value in zip(indices, values):
                    scores[i] = value
        finally:
            self.train(training)
        return scores

    def evaluate_batch(self, smiles_list: list[str]) -> list[float]:
        return [
            max(0.0, 100.0 + value)
            for value in self.log_likelihood(smiles_list, canonical=False)
        ]

    def _calc_scores_batch(
        self, seq: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        return (self.calc_p_x(seq, self(seq, lengths)) + 100).clamp(min=0)

    @torch.no_grad()
    def screening(
        self,
        smiles_list: list[str],
        naive: bool = False,
        batch_size: int = 64,
        verbose: bool = False,
    ) -> list[float]:
        """Legacy display scoring: minimum across enumerated isomers.

        naive=True uses the first enumerated isomer, as in deepdl. For canonical
        input with unspecified stereo preserved, use log_likelihood instead.
        Inputs are globally sorted by character length before batching, matching
        DeepDL screening; returned scores retain the original input order.
        """
        minimum = [math.inf] * len(smiles_list)
        pending: list[tuple[int, str]] = []

        def flush() -> None:
            values = self.log_likelihood(
                [s for _, s in pending], batch_size, canonical=False
            )
            for (index, _), value in zip(pending, values):
                minimum[index] = min(minimum[index], value)
            pending.clear()

        order = sorted(
            range(len(smiles_list)),
            key=lambda i: (len(smiles_list[i]), smiles_list[i]),
            reverse=True,
        )
        for index in tqdm(order, disable=not verbose, desc="screening"):
            smiles = smiles_list[index]
            mol = Chem.MolFromSmiles(smiles)
            isomers = EnumerateStereoisomers(mol)
            strings = (
                [Chem.MolToSmiles(next(isomers))]
                if naive
                else [Chem.MolToSmiles(m) for m in isomers]
            )
            for string in strings:
                pending.append((index, string))
                if len(pending) == batch_size:
                    flush()
        if pending:
            flush()
        return [max(0.0, 100.0 + value) for value in minimum]

    def save_pretrained(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.config.to_dict(),
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        device: Union[torch.device, str] = "cpu",
        config: Optional[DeepDL2Config] = None,
        token_path: Union[str, Path] = VOCAB_PATH,
    ) -> "DeepDL2_Base":
        checkpoint = torch.load(
            pretrained_model_name_or_path, map_location="cpu", weights_only=True
        )
        config = config or DeepDL2Config(**checkpoint["config"])
        model = cls.from_config(config, SmilesTokenizer(token_path))
        state = {
            key.removeprefix("model."): value
            for key, value in checkpoint["state_dict"].items()
        }
        model.load_state_dict(state)
        return model.to(device).eval()


class DeepDL2_TF(DeepDL2_Base):
    def __init__(
        self,
        config: Optional[DeepDL2Config] = None,
        tokenizer: Optional[SmilesTokenizer] = None,
    ) -> None:
        super().__init__(config or DeepDL2Config(), tokenizer)
        c = self.config
        self.transformer: TransformerStack = TransformerStack(
            c.hidden_size, c.n_heads, c.n_layers
        )
        self.fc: nn.Linear = nn.Linear(c.hidden_size, c.input_size, bias=False)
        self.apply(self._init_transformer)
        self.fc.weight = self.embedding.weight

    @staticmethod
    def _init_transformer(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self, seq: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        bos = seq.new_full((seq.size(0), 1), self.bos_token_id)
        x = self.embedding(torch.cat((bos, seq[:, :-1]), dim=1))
        return self.fc(self.transformer(x))


class DeepDL2_GRU(DeepDL2_Base):
    def __init__(
        self,
        config: Optional[DeepDL2Config] = None,
        tokenizer: Optional[SmilesTokenizer] = None,
    ) -> None:
        super().__init__(config or DeepDL2Config.preset("gru-medium"), tokenizer)
        c = self.config
        self.start_codon: nn.Parameter = nn.Parameter(torch.zeros(1, 1, c.hidden_size))
        self.GRU: nn.GRU = nn.GRU(
            c.hidden_size,
            c.hidden_size,
            c.n_layers,
            batch_first=True,
            dropout=c.dropout if c.n_layers > 1 else 0.0,
        )
        self.fc: nn.Linear = nn.Linear(c.hidden_size, c.input_size)

    def forward(
        self, seq: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.embedding(seq[:, :-1])
        x = torch.cat((self.start_codon.expand(seq.size(0), 1, -1), x), dim=1)
        if lengths is None:
            lengths = seq.ne(self.pad_token_id).sum(-1)
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        output, _ = self.GRU(packed)
        x, _ = pad_packed_sequence(output, batch_first=True, total_length=seq.size(1))
        return self.fc(x)


DeepDL2 = DeepDL2_Base
