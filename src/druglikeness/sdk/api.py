from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar, Union

import torch
import torch.nn as nn
from typing_extensions import Self


@dataclass
class ModelConfig: ...


_ModelConfigT = TypeVar("_ModelConfigT", bound=ModelConfig)


class DrugLikenessClient(nn.Module, ABC, Generic[_ModelConfigT]):
    # ==== Property ==== #
    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    # ==== Interface ==== #
    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        device: Union[torch.device, str] = "cpu",
        config: Optional[_ModelConfigT] = None,
    ) -> Self:
        """Returns a pretrained model instance.
        The output model is ready for inference. (eval-mode)
        """

    def scoring(self, smiles: str, naive: bool = False) -> float:
        """Returns a score for a single SMILES string."""
        return self.screening([smiles], naive, verbose=False)[0]

    @torch.no_grad()
    @abstractmethod
    def screening(
        self,
        smiles_list: list[str],
        naive: bool = False,
        batch_size: int = 64,
        verbose: bool = False,
    ) -> list[float]:
        """Returns a list of scores for a list of SMILES strings."""

    # ==== Model running ==== #
    @abstractmethod
    def forward(self, seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq: [L, B] where B is the batch size and L is the length of SMILES.
            lengths: [B,] where B is the batch size.
        Returns:
            logits: [L, B, C] where C is the number of characters.
        """

    @abstractmethod
    def calc_p_x(self, seq: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq: [L, B] where B is the batch size and L is the length of SMILES.
            logits: [L, B, C] where C is the number of characters.
        Returns:
            log_p_seq: [B,]
        """
