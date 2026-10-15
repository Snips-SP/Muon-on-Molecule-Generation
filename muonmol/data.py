"""Data loading for training and evaluation (PLAN.md stage LOOP; D13, D15, D30, D31).

Reads the DATA stage outputs (scripts/data_prep.py) from <data-root>/moses/processed/:

    vocab.json                                   character vocab, ids 0..3 = <pad> <bos> <eos> <unk>
    {train,test,test_scaffolds}_tokens.npy       uint8, all molecules concatenated, no special tokens
    {train,test,test_scaffolds}_offsets.npy      int64, molecule i = tokens[offsets[i]:offsets[i+1]]
    split_{val,10k,100k,full}.npy                int64 row indices into the train split (D15)

Built on torch.utils.data (Dataset, RandomSampler, BatchSampler, DataLoader) and
torch.nn.utils.rnn.pad_sequence.

Sequence format (D20, D30 = O13 option 2):

    full sequence  = [BOS] + characters + [EOS], right-padded with PAD to the longest molecule IN THE BATCH
    input_ids      = full[:, :-1]
    targets        = full[:, 1:], PAD replaced by IGNORE_INDEX (drops out of the loss)

A molecule with L characters has L + 1 input positions and L + 1 predicted tokens (characters + EOS).
In a shorter molecule's row the input also holds its EOS (and PAD) after position L; their targets are
IGNORE_INDEX and, with causal attention, no real position can see them (same convention as model_check).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler

IGNORE_INDEX = -100  # must equal muonmol.model.IGNORE_INDEX (checked in muonmol.train)
TRAIN_SCALES = ("10k", "100k", "full")  # D15
EVAL_SETS = ("val", "test", "test_scaffolds")  # val is carved from MOSES train (D15)
SEED_STRIDE = 1_000_003  # epoch generator seed = data_seed · SEED_STRIDE + epoch


@dataclass
class Batch:
    input_ids: torch.Tensor   # [B, T] int64
    targets: torch.Tensor     # [B, T] int64, IGNORE_INDEX at padded positions
    rows: torch.Tensor        # [B] row indices into the MOSES split (for checks)
    n_molecules: int          # B
    n_tokens: int             # predicted (non-padding) tokens = Σ (L_i + 1)
    n_positions: int          # B × T, incl. padding (compute actually spent)


class MolData:
    """All processed MOSES arrays plus the vocab. Loads lazily per split."""

    def __init__(self, data_root: Path | str):
        self.dir = Path(data_root).expanduser() / "moses" / "processed"
        if not self.dir.is_dir():
            raise FileNotFoundError(f"{self.dir} not found; run scripts/data_prep.py first")
        self.itos: list[str] = json.loads((self.dir / "vocab.json").read_text(encoding="utf-8"))["itos"]
        self.pad_id = self.itos.index("<pad>")
        self.bos_id = self.itos.index("<bos>")
        self.eos_id = self.itos.index("<eos>")
        self.meta: dict = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        self._arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._bos = torch.tensor([self.bos_id])
        self._eos = torch.tensor([self.eos_id])

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def arrays(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        """(tokens, offsets) of a MOSES split: train, test or test_scaffolds."""
        if split not in self._arrays:
            self._arrays[split] = (np.load(self.dir / f"{split}_tokens.npy"),
                                   np.load(self.dir / f"{split}_offsets.npy"))
        return self._arrays[split]

    def rows(self, name: str) -> tuple[str, np.ndarray]:
        """Which MOSES split a named set lives in, and its row indices.
        Training scales (10k, 100k, full) and val are row subsets of train (D15);
        test and test_scaffolds are whole MOSES splits."""
        if name in TRAIN_SCALES or name == "val":
            return "train", np.load(self.dir / f"split_{name}.npy")
        if name in ("test", "test_scaffolds"):
            _, offsets = self.arrays(name)
            return name, np.arange(len(offsets) - 1, dtype=np.int64)
        raise ValueError(f"unknown set {name!r}; expected one of {TRAIN_SCALES + EVAL_SETS}")

    def collate(self, items: list[tuple[int, torch.Tensor]]) -> Batch:
        """[(row, characters)] → padded batch (pad to the longest in the batch, D30)."""
        rows = torch.tensor([r for r, _ in items])
        seqs = [torch.cat([self._bos, chars, self._eos]) for _, chars in items]
        full = pad_sequence(seqs, batch_first=True, padding_value=self.pad_id)
        input_ids = full[:, :-1]
        targets = full[:, 1:].clone()
        targets[targets == self.pad_id] = IGNORE_INDEX
        n_tokens = sum(len(s) - 1 for s in seqs)
        return Batch(input_ids, targets, rows, len(seqs), n_tokens, input_ids.numel())

    def make_batch(self, split: str, rows) -> Batch:
        ds = MolSet(self, split, np.asarray(rows, dtype=np.int64))
        return self.collate([ds[i] for i in range(len(ds))])

    def eval_batches(self, name: str, batch_size: int) -> list[Batch]:
        """All molecules of a set, sorted by length (less padding; the summed loss does not depend
        on the order), in batches of at most batch_size."""
        split, rows = self.rows(name)
        _, offsets = self.arrays(split)
        lengths = offsets[rows + 1] - offsets[rows]
        rows = rows[np.argsort(lengths, kind="stable")]
        loader = DataLoader(MolSet(self, split, rows), batch_size=batch_size, shuffle=False, collate_fn=self.collate)
        return list(loader)


class MolSet(Dataset):
    """Molecules of one MOSES split, restricted to `rows`. Item i = (row, characters as int64 tensor)."""

    def __init__(self, data: MolData, split: str, rows: np.ndarray):
        self.tokens, self.offsets = data.arrays(split)
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[int, torch.Tensor]:
        r = int(self.rows[i])
        return r, torch.from_numpy(self.tokens[self.offsets[r]:self.offsets[r + 1]].astype(np.int64))


class BatchStream:
    """Endless training batches from one data scale.

    Each epoch: RandomSampler with a torch.Generator seeded from (data_seed, epoch), then
    BatchSampler(batch_size, drop_last=True). Every step therefore has exactly batch_size molecules;
    the n mod batch_size molecules left over are skipped in that epoch (a different set each epoch,
    10k scale: 16 of 10,000). data_seed defaults to the run's seed (D31), so both arms of one seed
    see the same batches.

    State = (epoch, pos = batches already used in this epoch); resume rebuilds the epoch's batch
    list and continues at pos, so it gives the same batches as an uninterrupted run.
    """

    def __init__(self, data: MolData, scale: str, batch_size: int, data_seed: int):
        if scale not in TRAIN_SCALES:
            raise ValueError(f"scale must be one of {TRAIN_SCALES}")
        self.data = data
        split, rows = data.rows(scale)
        self.dataset = MolSet(data, split, rows)
        self.batch_size = batch_size
        self.data_seed = data_seed
        self.steps_per_epoch = len(rows) // batch_size
        assert self.steps_per_epoch >= 1, "batch size larger than the data scale"
        self.epoch, self.pos = 0, 0
        self._start_epoch()

    @property
    def n_molecules(self) -> int:
        return len(self.dataset)

    def epoch_batches(self, epoch: int) -> list[list[int]]:
        """Dataset indices of every batch in an epoch."""
        g = torch.Generator().manual_seed(self.data_seed * SEED_STRIDE + epoch)
        return list(BatchSampler(RandomSampler(self.dataset, generator=g), self.batch_size, drop_last=True))

    def _start_epoch(self) -> None:
        batches = self.epoch_batches(self.epoch)[self.pos:]
        self._it = iter(DataLoader(self.dataset, batch_sampler=batches, collate_fn=self.data.collate))

    def next_batch(self) -> Batch:
        try:
            batch = next(self._it)
        except StopIteration:
            self.epoch, self.pos = self.epoch + 1, 0
            self._start_epoch()
            batch = next(self._it)
        self.pos += 1
        return batch

    @property
    def epochs_done(self) -> float:
        return self.epoch + self.pos / self.steps_per_epoch

    def state_dict(self) -> dict:
        return {"epoch": self.epoch, "pos": self.pos, "data_seed": self.data_seed,
                "batch_size": self.batch_size, "n_rows": len(self.dataset)}

    def load_state_dict(self, sd: dict) -> None:
        assert sd["data_seed"] == self.data_seed and sd["n_rows"] == len(self.dataset) \
            and sd["batch_size"] == self.batch_size, "data stream does not match the checkpoint"
        self.epoch, self.pos = sd["epoch"], sd["pos"]
        self._start_epoch()
