"""Transformer-based drug-likeness scoring."""

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import torch
from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from typing_extensions import Self

from druglikeness.sdk.api import DrugLikenessClient, ModelConfig

from .cleaning import ParentCleaner
from .tokenizer import SmilesTokenizer
from .transformer import TransformerStack


@dataclass
class ChemPriorConfig(ModelConfig):
    input_size: int = field(default_factory=lambda: len(SmilesTokenizer()))
    hidden_size: int = 384
    n_layers: int = 8
    n_heads: int = 6

    @classmethod
    def preset(cls, name: str = "medium", **overrides: Any) -> Self:
        """Return a model configuration for the specified size."""
        presets = {
            "small": dict(hidden_size=256, n_layers=8, n_heads=4),
            "medium": dict(hidden_size=384, n_layers=8, n_heads=6),
            "large": dict(hidden_size=512, n_layers=12, n_heads=8),
        }
        return cls(**{**presets[name], **overrides})

    def to_dict(self) -> dict[str, Any]:
        """Return the model configuration as a dictionary."""
        return asdict(self)


class ChemPrior(DrugLikenessClient[ChemPriorConfig]):
    """Autoregressive SMILES model for drug-likeness scoring."""

    def __init__(self, config: ChemPriorConfig) -> None:
        super().__init__()
        self.config: ChemPriorConfig = config
        self.tokenizer: SmilesTokenizer = SmilesTokenizer()
        self.vocab: dict[str, int] = self.tokenizer.token_to_id
        self.pad_token_id: int = self.tokenizer.pad_token_id
        self.bos_token_id: int = self.tokenizer.bos_token_id
        self.eos_token_id: int = self.tokenizer.eos_token_id
        c = self.config
        self.embedding: nn.Embedding = nn.Embedding(c.input_size, c.hidden_size)
        self.transformer: TransformerStack = TransformerStack(
            c.hidden_size, c.n_heads, c.n_layers
        )
        self.fc: nn.Linear = nn.Linear(c.hidden_size, c.input_size, bias=False)
        self.apply(self._init_weight)
        self.fc.weight = self.embedding.weight

    @property
    def device(self) -> torch.device:
        return self.embedding.weight.device

    # === Interface === #
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        device: Union[torch.device, str] = "cpu",
        config: Optional[ChemPriorConfig] = None,
    ) -> Self:
        """Load model weights and return a model ready for inference."""
        checkpoint = torch.load(
            pretrained_model_name_or_path, map_location="cpu", weights_only=True
        )
        if config is None:
            config = ChemPriorConfig(**checkpoint["config"])

        model = cls(config)
        state = {
            key.removeprefix("model."): value
            for key, value in checkpoint["state_dict"].items()
        }
        model.load_state_dict(state)
        return model.to(device).eval()

    @torch.no_grad()
    def screening(
        self,
        smiles_list: list[str],
        batch_size: int = 64,
        *,
        naive: bool = False,
        normalize: bool = True,
        verbose: bool = False,
    ) -> list[float]:
        """Return the lowest score across stereoisomers for each SMILES.

        Args:
            smiles_list: SMILES strings to score.
            batch_size: Number of stereoisomers evaluated per batch.
            naive: Evaluate only the first stereoisomer.
            normalize: Return 0–1 scores; otherwise return summed log probabilities.
            verbose: Display screening progress.
        Returns:
            Scores in the same order as the input SMILES.
            Invalid or empty SMILES receive 0.0, or -inf when normalize=False.
        Raises:
            ValueError: An input cannot be cleaned to one parent.
        """
        minimum = [math.inf] * len(smiles_list)
        cleaner = ParentCleaner()
        pending: list[tuple[int, str]] = []

        def flush() -> None:
            values = self.evaluate_batch([s for _, s in pending], normalize=normalize)
            for (index, _), value in zip(pending, values):
                minimum[index] = min(minimum[index], value)
            pending.clear()

        order = sorted(
            range(len(smiles_list)),
            key=lambda i: (len(smiles_list[i]), smiles_list[i]),
            reverse=True,
        )
        for index in tqdm(order, disable=not verbose, desc="screening"):
            mol = cleaner.clean(smiles_list[index])
            if mol is None:
                minimum[index] = 0.0 if normalize else -math.inf
                continue
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
        return minimum

    # === Internal === #

    @staticmethod
    def _init_weight(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self, seq: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Predict each target token from the preceding tokens.

        Args:
            seq: Target token IDs, including EOS, with shape [B, L].
            lengths: Sequence lengths with shape [B]; unused by the Transformer.
        Returns:
            Logits with shape [B, L, C], where C is the vocabulary size.
        """
        bos = seq.new_full((seq.size(0), 1), self.bos_token_id)
        x = self.embedding(torch.cat((bos, seq[:, :-1]), dim=1))
        return self.fc(self.transformer(x))

    def calc_p_x(self, seq: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Calculate sequence log probabilities, including EOS and excluding PAD.

        Args:
            seq: Target token IDs with shape [B, L].
            logits: Prediction logits with shape [B, L, C].
        Returns:
            Summed log probabilities with shape [B].
        """
        logits = logits.float()
        # Exclude PAD and BOS from the prediction distribution.
        mask = torch.zeros(logits.size(-1), dtype=torch.bool, device=logits.device)
        mask[self.pad_token_id] = mask[self.bos_token_id] = True
        logits = logits.masked_fill(mask, -torch.inf)

        log_p_t = logits.log_softmax(-1)

        log_p_t = log_p_t.gather(-1, seq.unsqueeze(-1)).squeeze(-1)

        log_p_t = log_p_t.masked_fill(seq == self.pad_token_id, 0.0)
        log_p_seq = log_p_t.sum(dim=-1)
        return log_p_seq

    def tokenize(self, smiles: str, add_special_tokens: bool = True) -> list[int]:
        """Return token IDs, appending EOS when add_special_tokens is True."""
        ids = self.tokenizer.encode(smiles, add_special_tokens=False)
        return [*ids, self.eos_token_id] if add_special_tokens else ids

    def encode(self, input: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode SMILES as padded target sequences.

        Args:
            input: SMILES strings to encode.
        Returns:
            Token IDs with shape [B, L] on the model device and lengths [B] on CPU.
        """
        rows = [torch.tensor(self.tokenize(s), dtype=torch.long) for s in input]
        lengths = torch.tensor([len(row) for row in rows], dtype=torch.long)
        seq = pad_sequence(rows, batch_first=True, padding_value=self.pad_token_id)
        return seq.to(self.device), lengths

    @torch.no_grad()
    def evaluate_batch(
        self,
        smiles_list: list[str],
        normalize: bool = True,
    ) -> list[float]:
        """Score prepared SMILES without cleaning or stereoisomer enumeration.

        normalize=True returns 0–1 scores; False returns summed log probabilities.
        Use screening() or scoring() for unvalidated input.
        """
        if not smiles_list:
            return []
        training = self.training
        self.eval()
        try:
            seq, lengths = self.encode(smiles_list)
            return self._calc_scores_batch(seq, lengths, normalize=normalize).tolist()
        finally:
            self.train(training)

    def _calc_scores_batch(
        self,
        seq: torch.Tensor,
        lengths: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Calculate scores or summed log probabilities for encoded sequences."""
        logits = self(seq, lengths)
        likelihood = self.calc_p_x(seq, logits)
        if normalize:
            scores = (1 + likelihood / 100).clamp(0, 1)
        else:
            scores = likelihood
        return scores

    def save_pretrained(self, path: Union[str, Path]) -> None:
        """Save model configuration and weights to a file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.config.to_dict(),
                "state_dict": self.state_dict(),
            },
            path,
        )
