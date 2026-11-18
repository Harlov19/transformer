"""Tokenizers and SentencePiece model training."""
import os
import tempfile
from pathlib import Path

PAD, BOS, EOS, UNK = 0, 1, 2, 3
NUM_SPECIAL = 4
SPECIAL_NAMES = ["<pad>", "<bos>", "<eos>", "<unk>"]
SYNTHETIC_TASKS = ("copy", "reverse", "sort")

def _cut_at_eos(ids):
    out = []
    for i in ids:
        if i == EOS:
            break
        out.append(i)
    return out


class SyntheticTokenizer:
    """Symbols 0..num_symbols-1 written as space-separated integers."""

    def __init__(self, num_symbols: int):
        self.num_symbols = num_symbols
        self.vocab_size = NUM_SPECIAL + num_symbols

    def encode(self, text):
        return [int(t) + NUM_SPECIAL for t in text.split()]

    def decode(self, ids):
        return " ".join(str(i - NUM_SPECIAL) for i in _cut_at_eos(ids) if i >= NUM_SPECIAL)

    def tokens(self, ids):
        return [SPECIAL_NAMES[i] if i < NUM_SPECIAL else str(i - NUM_SPECIAL) for i in ids]


class SentencePieceTokenizer:
    def __init__(self, model_path):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))
        self.vocab_size = self.sp.get_piece_size()

    def encode(self, text):
        return self.sp.encode(text, out_type=int)

    def decode(self, ids):
        return self.sp.decode([i for i in _cut_at_eos(ids) if i >= NUM_SPECIAL])

    def tokens(self, ids):
        return [self.sp.id_to_piece(i) for i in ids]


def _train_spm(pairs, run_dir, vocab_size):
    import sentencepiece as spm
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for s, t in pairs:                       # one shared vocabulary for both languages
            f.write(s.replace("\n", " ") + "\n" + t.replace("\n", " ") + "\n")
        path = f.name
    spm.SentencePieceTrainer.train(
        input=path, model_prefix=str(Path(run_dir) / "spm"), vocab_size=vocab_size,
        model_type="bpe", character_coverage=1.0,
        pad_id=PAD, bos_id=BOS, eos_id=EOS, unk_id=UNK,
    )
    os.remove(path)


def load_tokenizer(cfg, run_dir):
    if cfg.task in SYNTHETIC_TASKS:
        return SyntheticTokenizer(cfg.num_symbols)
    return SentencePieceTokenizer(Path(run_dir) / "spm.model")
