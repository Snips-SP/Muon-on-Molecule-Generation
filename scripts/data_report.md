# DATA report (MOSES)

- Date: 2026-10-12T14:02:31
- Command: `/mnt/c/Users/mbrun/Documents/University/Master Thesis/Muon on Molucules Generation/scripts/data_prep.py --data-root ~/muonmol-work/data`
- Python 3.11.15, numpy 2.4.6, RDKit 2026.09.1
- Source: https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv
- SHA-256: `bb47a94d347afd476d3828b5e26dceeabc42a2d8cf92a791d00349f22fea0d8b` (matches our 2026-10-12 download)
- Seed: 0

## Checks

| Check | Status |
|---|---|
| Canonical check (RDKit) | DONE (see section) |
| Validation disjoint from scales, scales nested (asserts) | PASS |
| Round trip, train: 0 failures | PASS |
| Round trip, test sets: failures = molecules with unknown tokens | PASS |
| Saved arrays decode back to the CSV strings (spot check) | PASS |

## Split sizes (MOSES)

| Split | Molecules | Unique strings |
|---|---|---|
| train | 1,584,663 | 1,584,663 |
| test | 176,074 | 176,074 |
| test_scaffolds | 176,225 | 176,225 |

Exact-string overlap between splits:

| Pair | Shared strings |
|---|---|
| train ∩ test | 0 |
| train ∩ test_scaffolds | 0 |
| test ∩ test_scaffolds | 0 |

## Molecule length, train split only (characters = tokens, D13)

| Statistic | Characters |
|---|---|
| Molecules | 1,584,663 |
| Min | 15 |
| Mean | 35.86 |
| Median | 36 |
| p90 | 42 |
| p99 | 46 |
| p99.9 | 49 |
| Max | 57 |

| Length bin | Molecules | Share |
|---|---|---|
| 15–19 | 269 | 0.0% |
| 20–24 | 8,090 | 0.5% |
| 25–29 | 130,640 | 8.2% |
| 30–34 | 469,178 | 29.6% |
| 35–39 | 616,651 | 38.9% |
| 40–44 | 324,654 | 20.5% |
| 45–49 | 34,192 | 2.2% |
| 50–54 | 978 | 0.1% |
| 55–59 | 11 | 0.0% |

## Padding waste (input for O13)

Share of padded positions. Positions per molecule = characters + 1 ([BOS] + chars as input, chars + [EOS] as target). Attention cost (∝ T²) is not included.

| Option | Context | Waste |
|---|---|---|
| 1: pad to longest train molecule | T = 58 | 36.5% |
| 2: pad to longest in batch (random batches, B = 32) | varies | 19.5% |
| 2: pad to longest in batch (random batches, B = 128) | varies | 23.0% |
| 2: pad to longest in batch (random batches, B = 512) | varies | 25.9% |

Option 2 rows are illustrative: the batch size is not decided (O3), and sorting batches by length would lower the waste further.

Test-set fit for option 1 (structural check only):

| Split | Max characters | Molecules longer than train max (57) |
|---|---|---|
| test | 55 | 0 |
| test_scaffolds | 56 | 0 |

## Data scales and validation (D15)

| Set | Molecules | Characters | Predicted tokens (chars + EOS) | N at D ≈ 20N (D9, my calc) |
|---|---|---|---|---|
| val | 10,000 | 357,846 | 367,846 | 18,392 |
| 10k | 10,000 | 359,064 | 369,064 | 18,453 |
| 100k | 100,000 | 3,583,512 | 3,683,512 | 184,176 |
| full | 1,574,663 | 56,466,278 | 58,040,941 | 2,902,047 |

The last column only feeds O1 (data-scale design); it is not a decision. D9 counts non-embedding parameters.

## Vocabulary (character-level, built from the full MOSES train split)

- Size: 30 (4 special + 26 characters)
- Special tokens: `<pad>`, `<bos>`, `<eos>`, `<unk>` (ids 0–3)

| Id | Character | Count in train | Molecules in train containing it |
|---|---|---|---|
| 4 | `#` | 127,115 | 122,407 |
| 5 | `(` | 5,460,725 | 1,581,090 |
| 6 | `)` | 5,460,725 | 1,581,090 |
| 7 | `-` | 295,097 | 272,887 |
| 8 | `1` | 4,455,986 | 1,584,079 |
| 9 | `2` | 2,771,422 | 1,234,452 |
| 10 | `3` | 829,620 | 403,859 |
| 11 | `4` | 88,622 | 44,140 |
| 12 | `5` | 2,024 | 1,011 |
| 13 | `6` | 14 | 7 |
| 14 | `=` | 2,259,380 | 1,409,276 |
| 15 | `B` | 52,677 | 52,081 |
| 16 | `C` | 10,726,584 | 1,579,578 |
| 17 | `F` | 491,767 | 298,718 |
| 18 | `H` | 179,011 | 169,153 |
| 19 | `N` | 2,346,896 | 1,455,374 |
| 20 | `O` | 3,305,613 | 1,509,346 |
| 21 | `S` | 285,476 | 279,512 |
| 22 | `[` | 179,011 | 169,153 |
| 23 | `]` | 179,011 | 169,153 |
| 24 | `c` | 14,222,955 | 1,557,824 |
| 25 | `l` | 187,351 | 166,921 |
| 26 | `n` | 2,332,619 | 1,024,964 |
| 27 | `o` | 255,480 | 238,498 |
| 28 | `r` | 52,677 | 52,081 |
| 29 | `s` | 276,266 | 261,718 |

Characters of the vocab that never occur in a data scale (embedding rows that get no training signal at that scale):

| Scale | Missing characters |
|---|---|
| 10k | `6` |
| 100k | none |
| full | none |

## Unknown tokens in test sets (needed again for atom-level tokens, O7)

| Split | Molecules with ≥ 1 unknown token | Share | Unknown characters |
|---|---|---|---|
| test | 0 | 0.0% | none |
| test_scaffolds | 0 | 0.0% | none |

## Round trip `decode(encode(s)) == s`

| Split | Failures | Expected |
|---|---|---|
| train | 0 | 0 |
| test | 0 | = molecules with unknown tokens |
| test_scaffolds | 0 | = molecules with unknown tokens |

## Canonical SMILES (RDKit)

RDKit 2026.09.1, `Chem.MolToSmiles(Chem.MolFromSmiles(s))` with default settings. The canonical form depends on the RDKit version.

| Split | Molecules | Already canonical | Not canonical | Invalid (RDKit cannot parse) | Time |
|---|---|---|---|---|---|
| train | 1,584,663 | 1,584,387 (100.0%) | 276 | 0 | 67 s |
| test | 176,074 | 176,038 (100.0%) | 36 | 0 | 8 s |
| test_scaffolds | 176,225 | 176,193 (100.0%) | 32 | 0 | 8 s |

Examples (original → canonical):

| Split | Original | Canonical |
|---|---|---|
| train | `O=c1c2ccccc2nc2n1-n1cnnc1-c1ccccc1-2` | `O=c1c2ccccc2nc2c3ccccc3c3nncn3n12` |
| train | `Cc1sc2nc3n(c(=O)c2c1C)-n1cnnc1-c1ccccc1-3` | `Cc1sc2nc3c4ccccc4c4nncn4n3c(=O)c2c1C` |
| train | `[H]N=c1c(C#N)c(C)c(C(=O)OCC)nn1-c1ccc(C)cc1` | `CCOC(=O)c1nn(-c2ccc(C)cc2)c(=N)c(C#N)c1C` |
| train | `[H]N=c1c(C#N)c(C)c(C(=O)OCC)nn1-c1ccc(C(C)(C)C)cc1` | `CCOC(=O)c1nn(-c2ccc(C(C)(C)C)cc2)c(=N)c(C#N)c1C` |
| train | `[H]N=c1c(C#N)c(C)c(C(=O)OCC)nn1-c1ccc(Cl)cc1` | `CCOC(=O)c1nn(-c2ccc(Cl)cc2)c(=N)c(C#N)c1C` |
| test | `[H]N=C1NC(=Nc2cccc(Cl)c2)NC(C)(C)N1` | `CC1(C)NC(=N)NC(=Nc2cccc(Cl)c2)N1` |
| test | `CC1(C)Cc2c(sc3c2-c2nnnn2-c2nnnn2-3)CO1` | `CC1(C)Cc2c(sc3c2c2nnnn2c2nnnn32)CO1` |
| test | `C1CCc2sc3c(c2CC1)-c1nnnn1-c1nnnn1-3` | `C1CCc2sc3c(c2CC1)c1nnnn1c1nnnn31` |
| test | `Cc1nnc2n1-c1nnc(C)n1-c1sc3c(c1-2)CCC(C)C3` | `Cc1nnc2c3c4c(sc3n3c(C)nnc3n12)CC(C)CC4` |
| test | `CSc1nnc2n1-c1sc3c(c1-c1ncnn1-2)C(C)CCC3` | `CSc1nnc2n3ncnc3c3c4c(sc3n12)CCCC4C` |
| test_scaffolds | `[H]N=c1scc(-c2ccc(C)cc2)n1-c1ccccc1` | `Cc1ccc(-c2csc(=N)n2-c2ccccc2)cc1` |
| test_scaffolds | `[H]N=c1scc(-c2ccc(C)cc2)n1-c1ccc(OC)cc1` | `COc1ccc(-n2c(-c3ccc(C)cc3)csc2=N)cc1` |
| test_scaffolds | `CC1(C)Cc2c(sc3c2-n2nnnc2-n2nnnc2-3)CO1` | `CC1(C)Cc2c(sc3c4nnnn4c4nnnn4c23)CO1` |
| test_scaffolds | `CCn1c(C(=O)OC)c(N)c2c1-c1ccccc1-n1nnnc1-2` | `CCn1c(C(=O)OC)c(N)c2c1c1ccccc1n1nnnc21` |
| test_scaffolds | `C=CCOC(=O)CCCc1nc2cccc3nc(C)nc(n1)-n-3-2` | `C=CCOC(=O)CCCC1=NC2=CC=CC3=NC(C)=NC(=N1)N23` |

## Output files

- `/home/mbrun/muonmol-work/data/moses/processed/meta.json`
- `/home/mbrun/muonmol-work/data/moses/processed/split_100k.npy`
- `/home/mbrun/muonmol-work/data/moses/processed/split_10k.npy`
- `/home/mbrun/muonmol-work/data/moses/processed/split_full.npy`
- `/home/mbrun/muonmol-work/data/moses/processed/split_val.npy`
- `/home/mbrun/muonmol-work/data/moses/processed/test_offsets.npy`
- `/home/mbrun/muonmol-work/data/moses/processed/test_scaffolds_offsets.npy`
- `/home/mbrun/muonmol-work/data/moses/processed/test_scaffolds_tokens.npy`
- `/home/mbrun/muonmol-work/data/moses/processed/test_tokens.npy`
- `/home/mbrun/muonmol-work/data/moses/processed/train_offsets.npy`
- `/home/mbrun/muonmol-work/data/moses/processed/train_tokens.npy`
- `/home/mbrun/muonmol-work/data/moses/processed/vocab.json`
- `/home/mbrun/muonmol-work/data/moses/raw/dataset_v1.csv`
