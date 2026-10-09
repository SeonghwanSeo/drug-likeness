"""Element/syntax SMILES tokenizer."""

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Union

from typing_extensions import Self

VOCAB_PATH: Path = Path(__file__).with_name("vocab.txt")

OUTER = re.compile(r"\[[^\[\]\s]+\]|Cl|Br|%\([0-9]+\)|%[0-9]{2}|->|<-|[\s\S]")
BRACKET = re.compile(r"\[([0-9]*)([A-Z][a-z]?|si|se|as|te|[bcnops]|\*)(.*)\]")
CHIRAL_CLASSES = {"TH", "AL", "SP", "TB", "OH"}


class SmilesTokenizer:
    def __init__(self, token_path: Union[str, Path] = VOCAB_PATH) -> None:
        self.tokens: tuple[str, ...] = tuple(Path(token_path).read_text().splitlines())
        self.token_to_id: dict[str, int] = {
            token: i for i, token in enumerate(self.tokens)
        }
        self.pad_token_id: int = self.token_to_id["<PAD>"]
        self.bos_token_id: int = self.token_to_id["<BOS>"]
        self.eos_token_id: int = self.token_to_id["<EOS>"]

    def __len__(self) -> int:
        return len(self.tokens)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> Self:
        return cls(path)

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text("\n".join(self.tokens) + "\n")

    def _split_bracket(self, text: str) -> list[str]:
        mass, element, tail = BRACKET.fullmatch(text).groups()
        parts = ["[", *mass, element]
        i = 0
        while i < len(tail):
            pair = tail[i : i + 2]
            if pair == "@@" or (i > 0 and tail[i - 1] == "@" and pair in CHIRAL_CLASSES):
                parts.append(pair)
                i += 2
            else:
                parts.append(tail[i])
                i += 1
        return parts + ["]"]

    def tokenize(self, smiles: str) -> list[str]:
        result: list[str] = []
        for match in OUTER.finditer(smiles):
            text = match.group()
            if text.startswith("["):
                result.extend(self._split_bracket(text))
            elif text.startswith("%"):
                result.extend(text)
            else:
                result.append(text)
        return result

    def encode(self, smiles: str, *, add_special_tokens: bool = True) -> list[int]:
        ids = [self.token_to_id[token] for token in self.tokenize(smiles)]
        return [self.bos_token_id, *ids, self.eos_token_id] if add_special_tokens else ids

    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        special = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        return "".join(
            self.tokens[i] for i in ids if not skip_special_tokens or i not in special
        )
