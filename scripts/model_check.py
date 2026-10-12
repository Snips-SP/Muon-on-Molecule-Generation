"""MODEL stage: checks for muonmol.model (PLAN.md stage MODEL; D20, D21).

Run inside the training environment (WSL2), from the repo root, after `pip install -e . --no-deps --no-build-isolation`:

    python scripts/model_check.py                     # all checks, report to scripts/model_report.md
    python scripts/model_check.py --device cpu        # dry run without the GPU
    python scripts/model_check.py --data-root ~/muonmol-work/data --steps 600 --lr 1e-3

Checks (required = must pass before any training run):

    | # | Check                                                          | Device      | Required |
    |---|----------------------------------------------------------------|-------------|----------|
    | 1 | Shapes and finite loss (separate and fused layouts)           | CPU fp32    | yes      |
    | 2 | Causality: changing token t leaves logits before t unchanged   | CPU fp32    | yes      |
    | 3 | Padding: a molecule gives the same logits alone and padded     | CPU fp32    | yes      |
    | 4 | Fused = separate layout with the same weights                  | CPU fp32    | yes      |
    | 5 | Same outputs as HF LlamaForCausalLM (eager and sdpa)           | CPU fp32    | yes      |
    | 6 | Parameter table, Muon vs AdamW share                           | –           | info     |
    | 7 | Overfit one batch down to its memorization floor              | --device, bf16 autocast | yes |
    | 8 | Complex numbers on the device                                  | --device    | info     |

Memorization floor (check 7): position 0 only sees [BOS], so the model cannot know which molecule
follows; the same holds wherever two molecules of the batch share a prefix. The lowest reachable
loss on a fixed batch is therefore the conditional entropy of the next token given the prefix,
computed exactly from the batch. "~0 loss" in PLAN.md means "reaches this floor".

The tolerances below are our own bounds (not from a source). Measured values are always reported.
Exit code: 0 if no required check failed, 1 otherwise.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import platform
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from muonmol.model import IGNORE_INDEX, LlamaLM, ModelConfig, param_groups, param_table

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

TOL_EXACT = 1e-6     # causality: logits before the changed token (should be identical)
TOL_LOGITS = 1e-5    # max |logit difference| (padding, fused vs separate, HF)
TOL_LOSS = 1e-6      # |loss difference| vs HF
TOL_GRAD = 1e-5      # max |grad diff| / max |grad|, per parameter, vs HF
FLOOR_MARGIN = 0.02  # overfit passes if final loss - memorization floor < this (nats per token)
HF_VERSION_READ = "5.19.0"  # transformers version whose modeling_llama.py the model mirrors

PASS, FAIL, INFO, SKIP = "PASS", "FAIL", "INFO", "SKIP"
RESULTS: list[dict] = []
CPU = torch.device("cpu")


def record(name: str, required: bool, status: str, details: list[str]) -> None:
    RESULTS.append({"name": name, "required": required, "status": status, "details": details})
    print(f"[{status}] {name}")
    for d in details:
        print(f"    {d}")


def run_check(name: str, required: bool, fn, *args) -> None:
    try:
        status, details = fn(*args)
    except Exception as e:  # a crashing check is a failed check, not a crashed script
        status, details = (FAIL if required else INFO), [f"Error: {type(e).__name__}: {e}",
                                                         "```", traceback.format_exc().strip(), "```"]
    record(name, required, status, details)


# ---------------------------------------------------------------------------
# Data: one real batch from the validation split (DATA stage outputs)
# ---------------------------------------------------------------------------

class Batch:
    def __init__(self, data_root: Path, n: int):
        proc = data_root / "moses" / "processed"
        itos = json.loads((proc / "vocab.json").read_text(encoding="utf-8"))["itos"]
        self.vocab_size = len(itos)
        self.pad, self.bos, self.eos = itos.index("<pad>"), itos.index("<bos>"), itos.index("<eos>")
        tokens = np.load(proc / "train_tokens.npy")
        offsets = np.load(proc / "train_offsets.npy")
        rows = np.load(proc / "split_val.npy")[:n]
        self.seqs = [[self.bos] + tokens[offsets[r]:offsets[r + 1]].tolist() + [self.eos] for r in rows]
        self.full = self.pad_to_longest(self.seqs)            # [B, L]: [BOS] + chars + [EOS] + PAD
        self.input_ids = self.full[:, :-1]                     # model input
        self.targets = self.full[:, 1:].clone()                # next-token targets
        self.targets[self.targets == self.pad] = IGNORE_INDEX

    def pad_to_longest(self, seqs: list[list[int]]) -> torch.Tensor:
        out = torch.full((len(seqs), max(map(len, seqs))), self.pad, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, :len(s)] = torch.tensor(s)
        return out


def memorization_floor(seqs: list[list[int]]) -> float:
    """Lowest reachable mean NLL (nats/token) on a fixed batch: H(next token | prefix)."""
    nxt: dict[tuple, Counter] = defaultdict(Counter)
    for s in seqs:
        for t in range(1, len(s)):
            nxt[tuple(s[:t])][s[t]] += 1
    total = sum(len(s) - 1 for s in seqs)
    h = 0.0
    for counts in nxt.values():
        n = sum(counts.values())
        h -= sum(c * math.log(c / n) for c in counts.values())
    return h / total


def new_model(cfg: ModelConfig, seed: int) -> LlamaLM:
    torch.manual_seed(seed)
    return LlamaLM(cfg)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_shapes(cfg: ModelConfig, b: Batch, seed: int):
    det, ok = [], True
    for fused in (False, True):
        c = ModelConfig(**{**cfg.to_dict(), "fused_qkv": fused, "fused_gate_up": fused})
        m = new_model(c, seed).eval()
        with torch.no_grad():
            logits, loss = m(b.input_ids, b.targets)
        want = (*b.input_ids.shape, c.vocab_size)
        good = tuple(logits.shape) == want and torch.isfinite(loss).item()
        ok &= good
        det.append(f"fused={fused}: logits {tuple(logits.shape)} (want {want}), loss {loss.item():.4f} "
                   f"(uniform guess ln V = {math.log(c.vocab_size):.4f})")
    return (PASS if ok else FAIL), det


def check_causality(cfg: ModelConfig, b: Batch, seed: int):
    m = new_model(cfg, seed).eval()
    x = b.input_ids[:4].clone()
    t = 10
    x2 = x.clone()
    x2[:, t] = (x2[:, t] % (cfg.vocab_size - 1)) + 1  # a different, non-PAD token
    with torch.no_grad():
        l1, _ = m(x)
        l2, _ = m(x2)
    before = (l1[:, :t] - l2[:, :t]).abs().max().item()
    at = (l1[:, t:] - l2[:, t:]).abs().max().item()
    ok = before <= TOL_EXACT and at > 0
    return (PASS if ok else FAIL), [f"Changed token at position {t}: max |Δlogit| before t = {before:.2e} "
                                    f"(bound {TOL_EXACT:.0e}), from t on = {at:.2e} (must be > 0)"]


def check_padding(cfg: ModelConfig, b: Batch, seed: int):
    m = new_model(cfg, seed).eval()
    lengths = [len(s) - 1 for s in b.seqs]  # input positions per molecule
    order = np.argsort(lengths)
    rows = [int(order[0]), int(order[len(order) // 2])]  # shortest (most padding) and a median one
    det, worst = [], 0.0
    with torch.no_grad():
        batch_logits, batch_nll = m(b.input_ids, b.targets, reduction="none")
        for i in rows:
            n = lengths[i]
            alone_logits, alone_nll = m(b.input_ids[i:i + 1, :n], b.targets[i:i + 1, :n], reduction="none")
            d = (alone_logits[0] - batch_logits[i, :n]).abs().max().item()
            dl = abs(alone_nll.sum().item() - batch_nll[i].sum().item())
            worst = max(worst, d)
            det.append(f"row {i}: {n} real positions, {b.input_ids.shape[1] - n} padded; "
                       f"max |Δlogit| = {d:.2e}, |Δ loss sum| = {dl:.2e}")
    pad_targets = int((b.targets == IGNORE_INDEX).sum())
    det.append(f"Padded target positions excluded from the loss: {pad_targets:,}")
    return (PASS if worst <= TOL_LOGITS else FAIL), det + [f"Bound: {TOL_LOGITS:.0e}"]


def check_fused(cfg: ModelConfig, b: Batch, seed: int):
    sep = new_model(cfg, seed).eval()
    fcfg = ModelConfig(**{**cfg.to_dict(), "fused_qkv": True, "fused_gate_up": True})
    fus = LlamaLM(fcfg).eval()
    fus.load_separate_state_dict(sep.separate_state_dict())
    with torch.no_grad():
        ls, loss_s = sep(b.input_ids, b.targets)
        lf, loss_f = fus(b.input_ids, b.targets)
    d = (ls - lf).abs().max().item()
    back = fus.separate_state_dict()
    roundtrip = all(torch.equal(back[k], v) for k, v in sep.separate_state_dict().items()) and \
        back.keys() == sep.separate_state_dict().keys()
    ok = d <= TOL_LOGITS and roundtrip
    return (PASS if ok else FAIL), [
        f"max |Δlogit| = {d:.2e} (bound {TOL_LOGITS:.0e}), |Δloss| = {abs(loss_s.item() - loss_f.item()):.2e}",
        f"fused → separate weights round trip exact: {roundtrip}",
        f"Fused matrices: qkv_proj {tuple(fus.model.layers[0].self_attn.qkv_proj.weight.shape)}, "
        f"gate_up_proj {tuple(fus.model.layers[0].mlp.gate_up_proj.weight.shape)}"]


def hf_config(cfg: ModelConfig):
    from transformers import LlamaConfig
    return LlamaConfig(
        vocab_size=cfg.vocab_size, hidden_size=cfg.hidden_size, intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_layers, num_attention_heads=cfg.num_heads, num_key_value_heads=cfg.num_heads,
        hidden_act="silu", max_position_embeddings=128, initializer_range=cfg.init_std,
        rms_norm_eps=cfg.rms_norm_eps, tie_word_embeddings=False,
        rope_parameters={"rope_type": "default", "rope_theta": cfg.rope_theta},
        attention_bias=False, mlp_bias=False, pad_token_id=None, bos_token_id=None, eos_token_id=None,
        use_cache=False)


def check_hf(cfg: ModelConfig, b: Batch, seed: int):
    import transformers
    from transformers import LlamaForCausalLM
    det = [f"transformers {transformers.__version__} (model mirrors {HF_VERSION_READ})"]
    ours = new_model(cfg, seed)
    sd = ours.separate_state_dict()
    labels = b.full.clone()
    labels[labels == b.pad] = IGNORE_INDEX
    ok = True
    for impl in ("eager", "sdpa"):
        hf = LlamaForCausalLM._from_config(hf_config(cfg), attn_implementation=impl)
        hf.load_state_dict(sd, strict=True)
        ours.zero_grad(set_to_none=True)
        hf.zero_grad(set_to_none=True)
        ours.eval(); hf.eval()  # no dropout anywhere; eval only fixes the mode
        logits, loss = ours(b.input_ids, b.targets)
        out = hf(input_ids=b.full, labels=labels, use_cache=False)
        hf_logits = out.logits[:, :-1]  # HF sees one more position (the last one has no target)
        d_logits = (logits - hf_logits).abs().max().item()
        d_loss = abs(loss.item() - out.loss.item())
        loss.backward()
        out.loss.backward()
        ours_p = dict(ours.named_parameters())
        worst, worst_name, n_params = 0.0, "", 0
        for name, p in hf.named_parameters():
            g_ref, g = p.grad, ours_p[name].grad
            rel = ((g - g_ref).abs().max() / g_ref.abs().max().clamp_min(1e-30)).item()
            n_params += 1
            if rel > worst:
                worst, worst_name = rel, name
        good = d_logits <= TOL_LOGITS and d_loss <= TOL_LOSS and worst <= TOL_GRAD and n_params == len(ours_p)
        ok &= good
        det.append(f"HF `{impl}`: max |Δlogit| = {d_logits:.2e}, |Δloss| = {d_loss:.2e} "
                   f"(ours {loss.item():.6f}, HF {out.loss.item():.6f}), worst relative grad diff "
                   f"{worst:.2e} in `{worst_name}` ({n_params} parameters compared) → {'ok' if good else 'NOT OK'}")
    # Initialization: our init vs a fresh HF init (same distribution expected, not the same numbers)
    torch.manual_seed(seed)
    hf_fresh = LlamaForCausalLM._from_config(hf_config(cfg), attn_implementation="sdpa")
    fresh = dict(hf_fresh.named_parameters())
    rows = []
    for name, p in ours.named_parameters():
        if name.startswith("model.layers.") and not name.startswith("model.layers.0."):
            continue  # one layer is enough
        q = fresh[name]
        rows.append(f"`{name}`: ours mean {p.mean().item():+.4f} std {p.std().item():.4f} | "
                    f"HF mean {q.mean().item():+.4f} std {q.std().item():.4f}")
    det += [f"Bounds: logits {TOL_LOGITS:.0e}, loss {TOL_LOSS:.0e}, relative grad {TOL_GRAD:.0e}",
            "Init (fresh models, layer 0 shown; expected std 0.02 for matrices, gains = 1):"] + rows
    return (PASS if ok else FAIL), det


def check_params(cfg: ModelConfig):
    tab = param_table(new_model(cfg, 0))
    tot = tab["total"]
    det = ["| Parameter | Shape | Matrices | Count | Group |", "|---|---|---|---|---|"]
    for kind, k in tab["kinds"].items():
        det.append(f"| `{kind}` | {k['shape']} | {k['n']} | {k['count']:,} | {k['group']} |")
    det += ["", "| Group | Parameters | Share |", "|---|---|---|"]
    for g, n in tab["per_group"].items():
        det.append(f"| {g} | {n:,} | {100 * n / tot:.2f}% |")
    det += [f"| total | {tot:,} | 100% |", "",
            f"Non-embedding parameters (without `embed_tokens`): {tab['non_embedding']:,}; "
            f"without `embed_tokens` and `lm_head`: {tab['non_embedding_non_head']:,}. Which of the two "
            f"D9 (Balaji, Chitsaz et al. 2026) means: [MISSING], check the paper.",
            "Fused layouts have the same counts (only the matrix shapes change)."]
    return INFO, det


def check_overfit(cfg: ModelConfig, b: Batch, seed: int, device: torch.device, steps: int, lr: float):
    m = new_model(cfg, seed).to(device).train()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    x, y = b.input_ids.to(device), b.targets.to(device)
    floor = memorization_floor(b.seqs)
    use_amp = device.type != "cpu"
    curve, t0 = [], time.time()
    for step in range(1, steps + 1):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            _, loss = m(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % max(1, steps // 10) == 0:
            curve.append((step, loss.item()))
    if device.type == "xpu":
        torch.xpu.synchronize()
    secs = time.time() - t0
    final = curve[-1][1]
    ok = math.isfinite(final) and final - floor < FLOOR_MARGIN
    det = [f"{len(b.seqs)} molecules, {int((y != IGNORE_INDEX).sum())} target tokens, AdamW lr {lr}, "
           f"betas (0.9, 0.95), no weight decay, {steps} steps, bf16 autocast: {use_amp}",
           f"Memorization floor of this batch: {floor:.4f} nats/token; final loss {final:.4f}; "
           f"gap {final - floor:.4f} (bound {FLOOR_MARGIN})",
           f"Time: {secs:.1f} s ({1000 * secs / steps:.1f} ms/step)",
           "Loss curve: " + ", ".join(f"{s}: {v:.3f}" for s, v in curve)]
    return (PASS if ok else FAIL), det


def check_complex(device: torch.device):
    torch.manual_seed(0)
    a = torch.randn(64, 2)
    ang = torch.rand(64) * 6.28
    ref = torch.view_as_real(torch.view_as_complex(a) * torch.polar(torch.ones(64), ang))
    out = torch.view_as_real(torch.view_as_complex(a.to(device)) *
                             torch.polar(torch.ones(64, device=device), ang.to(device))).cpu()
    d = (out - ref).abs().max().item()
    return INFO, [f"complex64 multiply (RoPE-style rotation) on {device}: max |Δ| vs CPU = {d:.2e}. "
                  "Info only: the model uses the real-valued HF form (D21)."]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def write_report(path: Path, header: list[str]) -> None:
    L = ["# MODEL check report", ""] + header + ["", "| # | Check | Required | Status |", "|---|---|---|---|"]
    for i, r in enumerate(RESULTS, 1):
        L.append(f"| {i} | {r['name']} | {'yes' if r['required'] else 'info'} | {r['status']} |")
    for i, r in enumerate(RESULTS, 1):
        L += ["", f"## {i}. {r['name']}: {r['status']}", ""]
        L += [d if d.startswith(("|", "```")) or d == "" else f"- {d}" for d in r["details"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="MODEL stage checks (see module docstring)")
    ap.add_argument("--data-root", type=Path, default=os.environ.get("MUONMOL_DATA"),
                    help="root folder for all datasets (default: $MUONMOL_DATA)")
    ap.add_argument("--device", default="xpu", help="device for checks 7 and 8 (default: xpu)")
    ap.add_argument("--report", type=Path, default=Path("scripts/model_report.md"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32, help="molecules in the test batch")
    ap.add_argument("--steps", type=int, default=600, help="steps for the overfit check")
    ap.add_argument("--lr", type=float, default=1e-3, help="AdamW LR for the overfit check")
    ap.add_argument("--no-hf", action="store_true", help="skip the HF comparison (NOT allowed before training, D20)")
    args = ap.parse_args()

    if args.data_root is None:
        print("ERROR: no data root. Pass --data-root or set MUONMOL_DATA (e.g. ~/muonmol-work/data).")
        return 1
    device = torch.device(args.device)
    if device.type == "xpu" and not torch.xpu.is_available():
        print("ERROR: xpu not available. Use --device cpu for a dry run.")
        return 1

    torch.set_default_dtype(torch.float32)
    b = Batch(Path(args.data_root).expanduser(), args.batch_size)
    cfg = ModelConfig(vocab_size=b.vocab_size)
    print(f"Config: {cfg.to_dict()}")

    run_check("Shapes and finite loss", True, check_shapes, cfg, b, args.seed)
    run_check("Causality", True, check_causality, cfg, b, args.seed)
    run_check("Padding", True, check_padding, cfg, b, args.seed)
    run_check("Fused = separate layout", True, check_fused, cfg, b, args.seed)
    if args.no_hf:
        record("Same outputs as HF LlamaForCausalLM", True, SKIP, ["Skipped (--no-hf). Must pass before training (D20)."])
    else:
        run_check("Same outputs as HF LlamaForCausalLM", True, check_hf, cfg, b, args.seed)
    run_check("Parameter table (Muon vs AdamW)", False, check_params, cfg)
    run_check("Overfit one batch", True, check_overfit, cfg, b, args.seed, device, args.steps, args.lr)
    run_check(f"Complex numbers on {device}", False, check_complex, device)

    header = [f"- Date: {datetime.datetime.now().isoformat(timespec='seconds')}",
              f"- Command: `{' '.join(sys.argv)}`",
              f"- Python {platform.python_version()}, PyTorch {torch.__version__}, device `{device}`",
              f"- Config: `{json.dumps(cfg.to_dict())}`",
              f"- Test batch: first {args.batch_size} molecules of the validation split (D15), seed {args.seed}"]
    write_report(args.report, header)
    print(f"Report written to {args.report}")
    failed = [r["name"] for r in RESULTS if r["required"] and r["status"] in (FAIL, SKIP)]
    if failed:
        print("Required checks not passed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
