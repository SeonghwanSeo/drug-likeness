# For SMILES
# fmt: off
CHARACTERS: list[str] = [
    "#", "%", "(", ")", "+", "-", "/", "0", "1", "2",
    "3", "4", "5", "6", "7", "8", "9", "=", "@", "A",
    "B", "C", "D", "E", "F", "G", "H", "I", "K", "L",
    "M", "N", "O", "P", "R", "S", "T", "U", "V", "W",
    "Y", "Z", "[", "\\", "]", "a", "b", "c", "d", "e",
    "f", "g", "h", "i", "k", "l", "m", "n", "o", "p",
    "r", "s", "t", "u", "y",
    # special tokens
    "Q", "_", 
]
# fmt: on
VOCAB: dict[str, int] = {c: i for i, c in enumerate(CHARACTERS)}
VOCAB_SIZE: int = len(VOCAB)  # 67

EOS_TOKEN: str = "Q"
EOS_TOKEN_ID: int = VOCAB[EOS_TOKEN]
PAD_TOKEN: str = "_"
PAD_TOKEN_ID: int = VOCAB[PAD_TOKEN]
