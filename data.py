"""Datasets and dataloaders.

Special ids are shared by every tokenizer: PAD=0, BOS=1, EOS=2, UNK=3.
Source sequences are  [tokens..., EOS].   Target sequences are  [BOS, tokens..., EOS].
"""
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from tokenization import (
    BOS,
    EOS,
    NUM_SPECIAL,
    PAD,
    SYNTHETIC_TASKS,
    SyntheticTokenizer,
    _train_spm,
    load_tokenizer,
)


@dataclass
class DataConfig:
    task: str = "reverse"            # copy | reverse | sort | translate
    # synthetic tasks
    num_symbols: int = 20
    syn_min_len: int = 3
    syn_max_len: int = 12
    n_train: int = 20000
    n_val: int = 1000
    n_test: int = 1000
    # translation
    dataset_name: str = "bentrevett/multi30k"
    src_lang: str = "de"
    tgt_lang: str = "en"
    spm_vocab_size: int = 8000
    # shared
    max_seq_len: int = 64
    batch_size: int = 64
    seed: int = 0

# datasets
class PairDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples  # list of (src LongTensor, tgt LongTensor)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def _build_example(src_ids, tgt_ids, max_len):
    src = src_ids[: max_len - 1] + [EOS]
    tgt = [BOS] + tgt_ids[: max_len - 2] + [EOS]
    return torch.tensor(src), torch.tensor(tgt)


def collate(batch):
    src, tgt = zip(*batch)
    return (pad_sequence(src, batch_first=True, padding_value=PAD),
            pad_sequence(tgt, batch_first=True, padding_value=PAD))


def make_loader(ds, batch_size, shuffle):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def _synthetic_split(cfg, n, rng):
    if cfg.syn_max_len + 2 > cfg.max_seq_len:
        raise ValueError("max_seq_len must be >= syn_max_len + 2")
    examples = []
    for _ in range(n):
        length = rng.randint(cfg.syn_min_len, cfg.syn_max_len)
        seq = [rng.randrange(cfg.num_symbols) + NUM_SPECIAL for _ in range(length)]
        tgt = {"copy": seq, "reverse": seq[::-1], "sort": sorted(seq)}[cfg.task]
        examples.append(_build_example(seq, tgt, cfg.max_seq_len))
    return PairDataset(examples)


def _load_hf_pairs(cfg):
    from datasets import load_dataset
    ds = load_dataset(cfg.dataset_name)
    split_names = {"train": "train", "val": "validation", "test": "test"}
    return {k: [(r[cfg.src_lang], r[cfg.tgt_lang]) for r in ds[v]] for k, v in split_names.items()}


def get_data(cfg: DataConfig, run_dir, splits=("train", "val", "test")):
    """Returns ({split: PairDataset}, tokenizer). Trains the BPE model on first use for translation."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    if cfg.task in SYNTHETIC_TASKS:
        for i, split in enumerate(("train", "val", "test")):
            if split in splits:
                rng = random.Random(cfg.seed * 1000 + i)
                out[split] = _synthetic_split(cfg, getattr(cfg, f"n_{split}"), rng)
        return out, SyntheticTokenizer(cfg.num_symbols)

    if cfg.task != "translate":
        raise ValueError(f"unknown task {cfg.task}")
    raw = _load_hf_pairs(cfg)
    if not (run_dir / "spm.model").exists():
        _train_spm(raw["train"], run_dir, cfg.spm_vocab_size)
    tok = load_tokenizer(cfg, run_dir)
    for split in splits:
        out[split] = PairDataset([
            _build_example(tok.encode(s), tok.encode(t), cfg.max_seq_len) for s, t in raw[split]
        ])
    return out, tok
