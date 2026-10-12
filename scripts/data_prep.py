"""DATA stage: MOSES data pipeline for the Muon vs AdamW project (see PLAN.md, stage DATA).

Run it inside the training environment (WSL2), from the repo root:

    export MUONMOL_DATA=~/muonmol-work/data              # once, e.g. in ~/.bashrc (data lives on the WSL disk)
    python scripts/data_prep.py                          # full run, report to scripts/data_report.md
    python scripts/data_prep.py --no-canonical-check     # skip the RDKit check (quick run)
    python scripts/data_prep.py --data-root ~/muonmol-work/data --report scripts/data_report.md --seed 0

What it does:

    | Step | What                                                                 | Decision |
    |------|----------------------------------------------------------------------|----------|
    | 1    | Download the MOSES CSV (if missing) and check its SHA-256            | D16      |
    | 2    | Statistics on the train split only: lengths, padding waste (O13)     | O13      |
    | 3    | Canonical-SMILES check with RDKit (all splits)                        | –        |
    | 4    | One seeded shuffle of train: validation + nested data scales         | D15      |
    | 5    | Character-level tokenizer, vocab from the train split, round trip     | D13      |
    | 6    | Pre-tokenize all splits, save arrays, write the Markdown report      | –        |

Outputs go to <data-root>/moses/ (outside the repo; everything can be rebuilt from this script + seed).
<data-root> comes from --data-root, otherwise from the environment variable MUONMOL_DATA:

    <data-root>/moses/raw/dataset_v1.csv
    <data-root>/moses/processed/vocab.json
    <data-root>/moses/processed/{train,test,test_scaffolds}_tokens.npy   uint8, all molecules concatenated, no special tokens
    <data-root>/moses/processed/{train,test,test_scaffolds}_offsets.npy  int64, molecule i = tokens[offsets[i]:offsets[i+1]]
    <data-root>/moses/processed/split_{val,10k,100k,full}.npy            int64 row indices into the train split (CSV order)
    <data-root>/moses/processed/meta.json

Sequence format used for all length numbers: [BOS] + characters + [EOS]. The model reads
[BOS] + characters and predicts characters + [EOS], so one molecule with L characters
uses L + 1 positions and has L + 1 predicted tokens. No special tokens are stored in the
token arrays; the data loader adds them, so the padding choice (O13) stays open.

Exit code: 0 if all checks pass, 1 otherwise.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import platform
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

MOSES_URL = "https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv"
# SHA-256 of the file as we downloaded it on 2026-10-12 (own download, not an official checksum).
EXPECTED_SHA256 = "bb47a94d347afd476d3828b5e26dceeabc42a2d8cf92a791d00349f22fea0d8b"
SPLITS = ("train", "test", "test_scaffolds")

N_VAL = 10_000                 # D15
SCALES = {"10k": 10_000, "100k": 100_000}  # plus "full" = all remaining train molecules (D15)

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIAL_TOKENS = (PAD, BOS, EOS, UNK)  # ids 0..3; PAD = 0

BATCH_SIZES_ILLUSTRATIVE = (32, 128, 512)  # batch size is not decided yet (O3)
N_EXAMPLES = 5  # examples shown per check in the report


# ---------------------------------------------------------------------------
# Tokenizer (character-level, D13)
# ---------------------------------------------------------------------------

class CharTokenizer:
    """One token per character. Ids 0..3 are the special tokens, then the sorted characters."""

    def __init__(self, chars: list[str]):
        self.itos: list[str] = list(SPECIAL_TOKENS) + sorted(chars)
        self.stoi: dict[str, int] = {s: i for i, s in enumerate(self.itos)}
        self.pad_id, self.bos_id, self.eos_id, self.unk_id = (self.stoi[t] for t in SPECIAL_TOKENS)
        assert self.pad_id == 0
        assert len(self.itos) <= 256, "uint8 token arrays need a vocab of at most 256"

    @classmethod
    def from_smiles(cls, smiles: list[str]) -> "CharTokenizer":
        chars = sorted(set("".join(smiles)))
        assert not set(chars) & set(SPECIAL_TOKENS)
        return cls(chars)

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        itos = json.loads(Path(path).read_text(encoding="utf-8"))["itos"]
        assert tuple(itos[: len(SPECIAL_TOKENS)]) == SPECIAL_TOKENS
        return cls(itos[len(SPECIAL_TOKENS):])

    def save(self, path: Path) -> None:
        data = {"type": "char", "special_tokens": list(SPECIAL_TOKENS), "itos": self.itos}
        Path(path).write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, s: str) -> list[int]:
        """Characters only, no BOS/EOS. Unknown characters become UNK."""
        return [self.stoi.get(c, self.unk_id) for c in s]

    def decode(self, ids) -> str:
        """Inverse of encode. Special tokens are dropped, except UNK, which is kept as '<unk>'."""
        out = []
        for i in ids:
            i = int(i)
            if i in (self.pad_id, self.bos_id, self.eos_id):
                continue
            out.append(self.itos[i])
        return "".join(out)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    log(f"Downloading {url}")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)


def read_moses(path: Path) -> dict[str, list[str]]:
    data: dict[str, list[str]] = {s: [] for s in SPLITS}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["SMILES", "SPLIT"], f"unexpected columns: {reader.fieldnames}"
        for row in reader:
            data[row["SPLIT"]].append(row["SMILES"])  # KeyError = unknown split name
    return data


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def md_table(header: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return lines


def md_code(s: str) -> str:
    return "`" + s.replace("|", "\\|") + "`"


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def length_stats(lengths: np.ndarray, seed: int) -> dict:
    """Train lengths (characters) -> distribution and padding waste (O13)."""
    pos = lengths + 1  # positions per molecule: [BOS] + chars as input, chars + [EOS] as target
    t_opt1 = int(pos.max())
    st = {
        "n": int(len(lengths)),
        "min": int(lengths.min()), "mean": float(lengths.mean()),
        "median": float(np.median(lengths)), "max": int(lengths.max()),
        "percentiles": {p: float(np.percentile(lengths, p)) for p in (90, 99, 99.9)},
        "context_T_option1": t_opt1,
        "waste_option1": float(1 - pos.mean() / t_opt1),
    }
    # Option 2 (illustrative): random batches, each padded to its own longest molecule.
    rng = np.random.default_rng(seed + 1)
    shuffled = pos[rng.permutation(len(pos))]
    st["waste_option2"] = {}
    for b in BATCH_SIZES_ILLUSTRATIVE:
        n = (len(shuffled) // b) * b
        batches = shuffled[:n].reshape(-1, b)
        st["waste_option2"][b] = float(1 - batches.sum() / (batches.max(axis=1).sum() * b))
    # Length histogram (5-character bins)
    lo = (st["min"] // 5) * 5
    bins = list(range(lo, st["max"] + 5, 5))
    counts, _ = np.histogram(lengths, bins=bins + [bins[-1] + 5])
    st["histogram"] = [(f"{a}–{a + 4}", int(c)) for a, c in zip(bins, counts) if c > 0]
    return st


def _canonical_one(s: str):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)  # RDKit defaults: canonical=True, isomericSmiles=True


def _init_worker() -> None:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")


def canonical_check(data: dict[str, list[str]], workers: int) -> dict:
    """Is each SMILES string already in RDKit's canonical form?"""
    import multiprocessing as mp
    import rdkit
    out = {"rdkit_version": rdkit.__version__, "splits": {}}
    with mp.Pool(workers, initializer=_init_worker) as pool:
        for split in SPLITS:
            t0 = time.time()
            smiles = data[split]
            canon = pool.map(_canonical_one, smiles, chunksize=2000)
            invalid = [s for s, c in zip(smiles, canon) if c is None]
            changed = [(s, c) for s, c in zip(smiles, canon) if c is not None and c != s]
            out["splits"][split] = {
                "n": len(smiles), "invalid": len(invalid), "non_canonical": len(changed),
                "canonical": len(smiles) - len(invalid) - len(changed),
                "examples_invalid": invalid[:N_EXAMPLES], "examples_changed": changed[:N_EXAMPLES],
                "seconds": time.time() - t0,
            }
            log(f"  canonical check {split}: {len(changed)} non-canonical, {len(invalid)} invalid")
    return out


def make_splits(n_train: int, seed: int) -> dict[str, np.ndarray]:
    """D15: one shuffle; first N_VAL = validation; the rest gives nested scales as prefixes."""
    perm = np.random.default_rng(seed).permutation(n_train).astype(np.int64)
    rest = perm[N_VAL:]
    splits = {"val": perm[:N_VAL]}
    for name, size in SCALES.items():
        splits[name] = rest[:size]
    splits["full"] = rest

    # Asserts (bugs, not report items)
    val, full = set(splits["val"].tolist()), set(splits["full"].tolist())
    assert len(val) == N_VAL and len(full) == n_train - N_VAL
    assert not val & full, "validation overlaps the training scales"
    assert val | full == set(range(n_train)), "some train molecules are in no split"
    prev = None
    for name in list(SCALES) + ["full"]:
        cur = set(splits[name].tolist())
        assert len(cur) == len(splits[name]), f"duplicate indices in {name}"
        if prev is not None:
            assert prev < cur, f"scales are not nested at {name}"
        prev = cur
    return splits


def tokenize_split(tok: CharTokenizer, smiles: list[str]) -> tuple[np.ndarray, np.ndarray, list]:
    """Encode all molecules, test decode(encode(s)) == s, return flat tokens, offsets, failures."""
    lengths = np.fromiter((len(s) for s in smiles), dtype=np.int64, count=len(smiles))
    offsets = np.zeros(len(smiles) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    tokens = np.empty(int(offsets[-1]), dtype=np.uint8)
    failures = []
    for i, s in enumerate(smiles):
        ids = tok.encode(s)
        if tok.decode(ids) != s:
            failures.append(s)
        tokens[offsets[i]:offsets[i + 1]] = ids
    return tokens, offsets, failures


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(path: Path, r: dict) -> None:
    L: list[str] = ["# DATA report (MOSES)", ""]
    L += [f"- Date: {r['date']}", f"- Command: `{r['command']}`",
          f"- Python {r['python']}, numpy {r['numpy']}, RDKit {r.get('rdkit_version', 'not used')}",
          f"- Source: {MOSES_URL}",
          f"- SHA-256: `{r['sha256']}` ({'matches' if r['sha256'] == EXPECTED_SHA256 else 'DOES NOT match'} our 2026-10-12 download)",
          f"- Seed: {r['seed']}", ""]

    L += ["## Checks", ""]
    L += md_table(["Check", "Status"], [[k, v] for k, v in r["checks"].items()]) + [""]

    L += ["## Split sizes (MOSES)", ""]
    L += md_table(["Split", "Molecules", "Unique strings"],
                  [[s, f"{r['counts'][s]:,}", f"{r['unique'][s]:,}"] for s in SPLITS]) + [""]
    L += ["Exact-string overlap between splits:", ""]
    L += md_table(["Pair", "Shared strings"], [[k, f"{v:,}"] for k, v in r["overlap"].items()]) + [""]

    st = r["length_stats"]
    L += ["## Molecule length, train split only (characters = tokens, D13)", ""]
    L += md_table(["Statistic", "Characters"],
                  [["Molecules", f"{st['n']:,}"], ["Min", st["min"]], ["Mean", f"{st['mean']:.2f}"],
                   ["Median", f"{st['median']:.0f}"]]
                  + [[f"p{p:g}", f"{v:.0f}"] for p, v in st["percentiles"].items()]
                  + [["Max", st["max"]]]) + [""]
    L += md_table(["Length bin", "Molecules", "Share"],
                  [[b, f"{c:,}", pct(c / st["n"])] for b, c in st["histogram"]]) + [""]

    L += ["## Padding waste (input for O13)", "",
          "Share of padded positions. Positions per molecule = characters + 1 ([BOS] + chars as input, "
          "chars + [EOS] as target). Attention cost (∝ T²) is not included.", ""]
    rows = [["1: pad to longest train molecule", f"T = {st['context_T_option1']}", pct(st["waste_option1"])]]
    rows += [[f"2: pad to longest in batch (random batches, B = {b})", "varies", pct(w)]
             for b, w in st["waste_option2"].items()]
    L += md_table(["Option", "Context", "Waste"], rows) + [""]
    L += ["Option 2 rows are illustrative: the batch size is not decided (O3), and sorting batches by "
          "length would lower the waste further.", ""]
    L += ["Test-set fit for option 1 (structural check only):", ""]
    L += md_table(["Split", "Max characters", f"Molecules longer than train max ({st['max']})"],
                  [[s, r["max_len"][s], f"{r['longer_than_train_max'][s]:,}"] for s in ("test", "test_scaffolds")]) + [""]

    L += ["## Data scales and validation (D15)", ""]
    rows = []
    for name, info in r["scales"].items():
        rows.append([name, f"{info['molecules']:,}", f"{info['chars']:,}", f"{info['predicted_tokens']:,}",
                     f"{info['predicted_tokens'] / 20:,.0f}"])
    L += md_table(["Set", "Molecules", "Characters", "Predicted tokens (chars + EOS)",
                   "N at D ≈ 20N (D9, my calc)"], rows) + [""]
    L += ["The last column only feeds O1 (data-scale design); it is not a decision. D9 counts "
          "non-embedding parameters.", ""]

    v = r["vocab"]
    L += ["## Vocabulary (character-level, built from the full MOSES train split)", "",
          f"- Size: {v['size']} ({len(SPECIAL_TOKENS)} special + {v['size'] - len(SPECIAL_TOKENS)} characters)",
          f"- Special tokens: {', '.join(md_code(t) for t in SPECIAL_TOKENS)} (ids 0–3)", ""]
    L += md_table(["Id", "Character", "Count in train", "Molecules in train containing it"],
                  [[i, md_code(c), f"{n:,}", f"{m:,}"] for i, c, n, m in v["chars"]]) + [""]
    L += ["Characters of the vocab that never occur in a data scale (embedding rows that get no "
          "training signal at that scale):", ""]
    L += md_table(["Scale", "Missing characters"],
                  [[k, ", ".join(md_code(c) for c in m) or "none"] for k, m in v["missing_in_scale"].items()]) + [""]

    L += ["## Unknown tokens in test sets (needed again for atom-level tokens, O7)", ""]
    L += md_table(["Split", "Molecules with ≥ 1 unknown token", "Share", "Unknown characters"],
                  [[s, f"{u['molecules']:,}", pct(u["molecules"] / r["counts"][s]),
                    ", ".join(md_code(c) for c in u["chars"]) or "none"]
                   for s, u in r["unknown"].items()]) + [""]

    L += ["## Round trip `decode(encode(s)) == s`", ""]
    L += md_table(["Split", "Failures", "Expected"],
                  [[s, f"{n:,}", "0" if s == "train" else "= molecules with unknown tokens"]
                   for s, n in r["roundtrip_failures"].items()]) + [""]

    L += ["## Canonical SMILES (RDKit)", ""]
    c = r.get("canonical")
    if c is None:
        L += ["Skipped (`--no-canonical-check`).", ""]
    else:
        L += [f"RDKit {c['rdkit_version']}, `Chem.MolToSmiles(Chem.MolFromSmiles(s))` with default "
              "settings. The canonical form depends on the RDKit version.", ""]
        L += md_table(["Split", "Molecules", "Already canonical", "Not canonical", "Invalid (RDKit cannot parse)", "Time"],
                      [[s, f"{x['n']:,}", f"{x['canonical']:,} ({pct(x['canonical'] / x['n'])})",
                        f"{x['non_canonical']:,}", f"{x['invalid']:,}", f"{x['seconds']:.0f} s"]
                       for s, x in c["splits"].items()]) + [""]
        ex = [(s, a, b) for s, x in c["splits"].items() for a, b in x["examples_changed"]]
        if ex:
            L += ["Examples (original → canonical):", ""]
            L += md_table(["Split", "Original", "Canonical"], [[s, md_code(a), md_code(b)] for s, a, b in ex]) + [""]
        inv = [(s, a) for s, x in c["splits"].items() for a in x["examples_invalid"]]
        if inv:
            L += ["Invalid examples:", ""] + md_table(["Split", "SMILES"], [[s, md_code(a)] for s, a in inv]) + [""]

    L += ["## Output files", ""] + [f"- `{p}`" for p in r["outputs"]] + [""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", type=Path, default=os.environ.get("MUONMOL_DATA"),
                    help="root folder for all datasets (default: $MUONMOL_DATA); MOSES goes to <data-root>/moses")
    ap.add_argument("--report", type=Path, default=Path("scripts/data_report.md"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-canonical-check", action="store_true", help="skip the RDKit check")
    ap.add_argument("--workers", type=int, default=4, help="processes for the RDKit check")
    ap.add_argument("--accept-new-hash", action="store_true",
                    help="continue although the CSV hash differs from our 2026-10-12 download")
    args = ap.parse_args()

    if args.data_root is None:
        print("ERROR: no data root. Pass --data-root or set MUONMOL_DATA (e.g. ~/muonmol-work/data).")
        return 1
    data_dir = Path(args.data_root).expanduser() / "moses"
    raw = data_dir / "raw" / "dataset_v1.csv"
    out = data_dir / "processed"
    out.mkdir(parents=True, exist_ok=True)
    r: dict = {"date": datetime.datetime.now().isoformat(timespec="seconds"),
               "command": " ".join(sys.argv), "python": platform.python_version(),
               "numpy": np.__version__, "seed": args.seed, "data_dir": str(data_dir), "checks": {}}

    # 1. Download + hash ----------------------------------------------------
    if not raw.exists():
        download(MOSES_URL, raw)
    r["sha256"] = sha256_of(raw)
    if r["sha256"] != EXPECTED_SHA256 and not args.accept_new_hash:
        print(f"ERROR: SHA-256 {r['sha256']} differs from our 2026-10-12 download "
              f"({EXPECTED_SHA256}). The data changed; check it, then rerun with --accept-new-hash.")
        return 1
    log("Reading CSV")
    data = read_moses(raw)
    r["counts"] = {s: len(data[s]) for s in SPLITS}
    sets = {s: set(data[s]) for s in SPLITS}
    r["unique"] = {s: len(sets[s]) for s in SPLITS}
    r["overlap"] = {"train ∩ test": len(sets["train"] & sets["test"]),
                    "train ∩ test_scaffolds": len(sets["train"] & sets["test_scaffolds"]),
                    "test ∩ test_scaffolds": len(sets["test"] & sets["test_scaffolds"])}
    del sets
    log(f"Counts: {r['counts']}")

    # 2. Train statistics + padding waste ------------------------------------
    train_len = np.fromiter((len(s) for s in data["train"]), dtype=np.int64)
    r["length_stats"] = length_stats(train_len, args.seed)
    r["max_len"] = {s: max(len(x) for x in data[s]) for s in SPLITS}
    r["longer_than_train_max"] = {s: sum(len(x) > r["length_stats"]["max"] for x in data[s])
                                  for s in ("test", "test_scaffolds")}
    log(f"Train length: mean {r['length_stats']['mean']:.2f}, max {r['length_stats']['max']}, "
        f"option-1 waste {pct(r['length_stats']['waste_option1'])}")

    # 3. Canonical check ------------------------------------------------------
    if args.no_canonical_check:
        r["checks"]["Canonical check (RDKit)"] = "SKIPPED"
    else:
        try:
            import rdkit  # noqa: F401
        except ImportError:
            print("ERROR: RDKit is not installed (pip install rdkit). Use --no-canonical-check to skip.")
            return 1
        log("Canonical check (RDKit)")
        r["canonical"] = canonical_check(data, args.workers)
        r["rdkit_version"] = r["canonical"]["rdkit_version"]
        r["checks"]["Canonical check (RDKit)"] = "DONE (see section)"

    # 4. Splits ----------------------------------------------------------------
    splits = make_splits(len(data["train"]), args.seed)
    r["checks"]["Validation disjoint from scales, scales nested (asserts)"] = "PASS"
    for name, idx in splits.items():
        np.save(out / f"split_{name}.npy", idx)

    # 5. Tokenizer + round trip ---------------------------------------------
    tok = CharTokenizer.from_smiles(data["train"])
    tok.save(out / "vocab.json")
    assert CharTokenizer.load(out / "vocab.json").itos == tok.itos
    char_count = Counter("".join(data["train"]))
    mol_count = Counter(c for s in data["train"] for c in set(s))
    r["vocab"] = {"size": len(tok),
                  "chars": [(tok.stoi[c], c, char_count[c], mol_count[c]) for c in tok.itos[len(SPECIAL_TOKENS):]]}

    # 6. Pre-tokenize -----------------------------------------------------------
    r["roundtrip_failures"], r["unknown"], arrays = {}, {}, {}
    for split in SPLITS:
        log(f"Tokenizing {split}")
        tokens, offsets, failures = tokenize_split(tok, data[split])
        np.save(out / f"{split}_tokens.npy", tokens)
        np.save(out / f"{split}_offsets.npy", offsets)
        arrays[split] = (tokens, offsets)
        r["roundtrip_failures"][split] = len(failures)
        if split != "train":
            unk_mask = tokens == tok.unk_id
            mol_has_unk = np.add.reduceat(unk_mask.astype(np.int64), offsets[:-1]) > 0
            unk_chars = sorted({c for s in data[split] for c in s if c not in tok.stoi})
            r["unknown"][split] = {"molecules": int(mol_has_unk.sum()), "chars": unk_chars}

    ok_train = r["roundtrip_failures"]["train"] == 0
    ok_test = all(r["roundtrip_failures"][s] == r["unknown"][s]["molecules"] for s in ("test", "test_scaffolds"))
    r["checks"]["Round trip, train: 0 failures"] = "PASS" if ok_train else "FAIL"
    r["checks"]["Round trip, test sets: failures = molecules with unknown tokens"] = "PASS" if ok_test else "FAIL"

    # Per-scale token counts and characters missing per scale
    tr_tokens, tr_off = arrays["train"]
    tr_len = np.diff(tr_off)
    r["scales"], r["vocab"]["missing_in_scale"] = {}, {}
    for name, idx in splits.items():
        chars = int(tr_len[idx].sum())
        r["scales"][name] = {"molecules": len(idx), "chars": chars, "predicted_tokens": chars + len(idx)}
        if name != "val":
            mol_mask = np.zeros(len(tr_len), dtype=bool)
            mol_mask[idx] = True
            present = set(np.unique(tr_tokens[np.repeat(mol_mask, tr_len)]).tolist())
            r["vocab"]["missing_in_scale"][name] = [c for c in tok.itos[len(SPECIAL_TOKENS):] if tok.stoi[c] not in present]

    # Check that a saved molecule decodes back from the saved arrays
    t = np.load(out / "train_tokens.npy"); o = np.load(out / "train_offsets.npy")
    i = int(splits["val"][0])
    assert tok.decode(t[o[i]:o[i + 1]]) == data["train"][i]
    r["checks"]["Saved arrays decode back to the CSV strings (spot check)"] = "PASS"

    # meta.json + report ---------------------------------------------------------
    r["outputs"] = sorted(str(p) for p in out.iterdir()) + [str(raw)]
    meta = {k: r[k] for k in ("date", "command", "python", "numpy", "seed", "sha256", "counts")}
    meta.update({"source_url": MOSES_URL, "vocab_size": len(tok), "n_val": N_VAL,
                 "scales": {k: v["molecules"] for k, v in r["scales"].items()},
                 "rdkit": r.get("rdkit_version"), "format": "flat uint8 tokens + int64 offsets, no special tokens"})
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    r["outputs"] = sorted(set(r["outputs"]) | {str(out / "meta.json")})
    write_report(args.report, r)
    log(f"Report written to {args.report}")

    failed = [k for k, v in r["checks"].items() if v == "FAIL"]
    for k, v in r["checks"].items():
        print(f"  {v:8s} {k}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
