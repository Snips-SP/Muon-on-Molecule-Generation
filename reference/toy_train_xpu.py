"""REF stage: adapted copy of Moonlight's examples/toy_train.py.

Original: https://github.com/MoonshotAI/Moonlight (commit c2ad5b2), MIT license, see
LICENSE_Moonlight. The unchanged original is in reference/toy_train.py.

Changes (see DECISIONS.md D10-D12):
    - Data: raw UTF-8 bytes instead of the Qwen2.5 tokenizer (vocab 257 = 256 bytes + 1
      end-of-document token). The last 1% of documents is held out for validation (D11).
    - Model: small Qwen2 (hidden 256, 4 layers, 4 heads, MLP 1024), fp32 as in the original.
    - New optimizer choice "muon_torch": torch.optim.Muon (match_rms_adamw) for 2D hidden
      weights + torch.optim.AdamW for embeddings, norms and biases, same grouping rule as the
      script. "adamw" and "muon_script" are the original choices (the script's Muon class and
      Newton-Schulz function have unchanged logic, including the @torch.compile on
      Newton-Schulz; only long docstrings were shortened).
    - Not used: trust_remote_code=True in load_dataset (not needed for this dataset).
    - --compile-opt: wraps torch.optim.Muon's step with torch.compile (D12, run 3).
    - --max-steps sets the cosine schedule; --stop-after ends training early with the same
      schedule (D12, run 4).
    - xpu device, fixed seed (same initial weights and batch order in every run), --wd is
      passed through, wandb logging, timing of the optimizer step, outputs in ~/muonmol-work.

Runs (D12), from the project root in WSL:
    0  python reference/toy_train_xpu.py --optimizer muon_torch --max-steps 50 --run-name trial
    0b python reference/toy_train_xpu.py --optimizer muon_torch --max-steps 50 --compile-opt --run-name trial_compiled
    1  python reference/toy_train_xpu.py --optimizer adamw       --max-steps N --run-name adamw
    2  python reference/toy_train_xpu.py --optimizer muon_torch  --max-steps N --run-name muon_torch
    3  python reference/toy_train_xpu.py --optimizer muon_torch  --max-steps N --compile-opt --run-name muon_torch_compiled
    4  python reference/toy_train_xpu.py --optimizer muon_script --max-steps N --stop-after 100 --run-name muon_script
"""

import argparse
import csv
import math
import os
import random
import time

import numpy as np
import torch
import torch._dynamo
from datasets import load_dataset
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from transformers import Qwen2Config, Qwen2ForCausalLM, get_cosine_schedule_with_warmup

EOS = 256          # end-of-document token (bytes use ids 0-255)
VOCAB_SIZE = 257


# ---------------------------------------------------------------------------
# Data: raw UTF-8 bytes
# ---------------------------------------------------------------------------

def encode_docs(texts, n_bytes_needed: int) -> torch.Tensor:
    """Concatenate documents as UTF-8 bytes + EOS until n_bytes_needed is reached."""
    chunks, total = [], 0
    for text in texts:
        arr = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int64)
        chunks.append(arr)
        chunks.append(np.array([EOS], dtype=np.int64))
        total += len(arr) + 1
        if total >= n_bytes_needed:
            break
    return torch.from_numpy(np.concatenate(chunks))


class BlockDataset(Dataset):
    """Cuts a long byte sequence into non-overlapping blocks of seq_len (as the original)."""

    def __init__(self, tokens: torch.Tensor, seq_len: int):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        return self.tokens[start:start + self.seq_len]


def get_data(args):
    name2path = {"openwebtext-100k": "Elriggs/openwebtext-100k"}
    texts = list(load_dataset(name2path[args.dataset], split="train")["text"])

    n_val_docs = max(1, int(len(texts) * args.val_frac))
    train_texts, val_texts = texts[:-n_val_docs], texts[-n_val_docs:]

    train_needed = args.max_steps * args.batch_size * args.seq_len + args.seq_len
    train_tokens = encode_docs(train_texts, train_needed)
    val_tokens = encode_docs(val_texts, args.val_blocks * args.seq_len)

    train_ds = BlockDataset(train_tokens, args.seq_len)
    val_ds = BlockDataset(val_tokens, args.seq_len)
    logger.info(f"Train: {len(train_tokens):,} bytes = {len(train_ds):,} blocks "
                f"(needed {train_needed:,}); validation: {len(val_ds)} blocks "
                f"from the first of {n_val_docs} held-out documents")
    if len(train_tokens) < train_needed:
        logger.warning("Not enough training data for one pass: blocks will be repeated")

    # Seeded generator: the same batch order in every run.
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              generator=g, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def get_model(args):
    # Same Qwen2 settings as the original, except the size and vocabulary (D10).
    config = Qwen2Config(
        attention_dropout=0.0,
        bos_token_id=EOS,
        eos_token_id=EOS,
        hidden_act="silu",
        hidden_size=args.hidden_size,
        initializer_range=0.02,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.seq_len + 1,
        max_window_layers=args.num_layers,
        model_type="qwen2",
        num_attention_heads=args.num_heads,
        num_hidden_layers=args.num_layers,
        num_key_value_heads=args.num_heads,
        rms_norm_eps=1e-06,
        rope_theta=1000000.0,
        sliding_window=1024,
        tie_word_embeddings=True,
        use_cache=False,
        use_mrope=False,
        use_sliding_window=False,
        vocab_size=VOCAB_SIZE,
    )
    return Qwen2ForCausalLM(config).float()  # fp32 as in the original


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------

# This code snippet is a modified version adapted from the following GitHub repository:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
# (Unchanged from Moonlight's toy_train.py.)
@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Moonlight's Muon (unchanged): Muon for muon_params, built-in AdamW for adamw_params."""

    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
    ):

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        # Sort parameters into those for which we will use Muon, and those for which we will not
        for p in muon_params:
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            # generate weight updates
            for p in params:
                # sanity check
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None

                # calc update
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                # scale update
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                # apply weight decay
                p.data.mul_(1 - lr * wd)

                # apply update
                p.data.add_(u, alpha=-adjusted_lr)

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss


def split_params(model):
    """The original grouping rule: 2D weights except embedding / LM head -> Muon, rest -> AdamW."""
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name:
            muon_params.append(p)
        else:
            adamw_params.append(p)
    return muon_params, adamw_params


def get_optimizers(name, model, lr, wd, compile_opt):
    """Returns a list of (label, optimizer). The training loop steps all of them."""
    if name == "adamw":  # original
        return [("adamw", torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd,
                                            betas=(0.9, 0.95)))]

    muon_params, adamw_params = split_params(model)
    if name == "muon_script":  # original
        return [("muon_script", Muon(lr=lr, wd=wd, muon_params=muon_params,
                                     adamw_params=adamw_params))]

    if name == "muon_torch":  # new: same settings as the script's Muon and its built-in AdamW
        # With compile, the LR is a tensor so that LR changes do not force a recompile.
        muon_lr = torch.tensor(lr) if compile_opt else lr
        muon = torch.optim.Muon(muon_params, lr=muon_lr, weight_decay=wd, momentum=0.95,
                                nesterov=True, ns_steps=5, adjust_lr_fn="match_rms_adamw")
        adamw = torch.optim.AdamW(adamw_params, lr=lr, weight_decay=wd, betas=(0.9, 0.95),
                                  eps=1e-8)
        return [("muon", muon), ("adamw", adamw)]

    raise ValueError(f"optimizer {name} not supported")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, val_loader, device) -> float:
    """Mean validation loss in bits per byte."""
    model.eval()
    losses = []
    for batch in val_loader:
        batch = batch.to(device)
        losses.append(model(input_ids=batch, labels=batch).loss.item())
    model.train()
    return sum(losses) / len(losses) / math.log(2)


def sync(device):
    if device.type == "xpu":
        torch.xpu.synchronize()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "muon_script", "muon_torch"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--dataset", default="openwebtext-100k")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--intermediate_size", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, required=True, help="length of the LR schedule")
    parser.add_argument("--stop-after", type=int, default=None, help="stop early, same schedule")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--val-blocks", type=int, default=100)
    parser.add_argument("--val-frac", type=float, default=0.01)
    parser.add_argument("--compile-opt", action="store_true", help="torch.compile the Muon step")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="xpu", choices=["xpu", "cpu"])
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    args = parser.parse_args()

    if args.compile_opt and args.optimizer != "muon_torch":
        parser.error("--compile-opt is only for --optimizer muon_torch")

    work_dir = os.path.join(os.path.expanduser(os.environ.get("MUONMOL_WORK", "~/muonmol-work")),
                            "ref", args.run_name)
    os.makedirs(work_dir, exist_ok=True)
    logger.add(os.path.join(work_dir, "train.log"))

    # Same seed -> same initial weights and same batch order in every run.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    train_loader, val_loader = get_data(args)
    model = get_model(args).to(device)
    model.train()

    optimizers = get_optimizers(args.optimizer, model, args.lr, args.wd, args.compile_opt)
    schedulers = [get_cosine_schedule_with_warmup(optimizer=opt, num_warmup_steps=args.warmup,
                                                  num_training_steps=args.max_steps, num_cycles=0.5)
                  for _, opt in optimizers]

    # Optimizer step functions (option A2: compile the whole Muon step).
    # The LR scheduler wraps opt.step in a call counter from lr_scheduler.py, a file that
    # torch.compile refuses to trace (graph break in the first compiled trial). So we compile
    # the class's own step function instead and call it with the optimizer. This skips only the
    # scheduler's counter; the scheduler may then warn once about the call order (harmless).
    # fullgraph=True: any graph break raises an error instead of silently running uncompiled.
    step_fns = []
    for label, opt in optimizers:
        if args.compile_opt and label == "muon":
            if hasattr(torch._dynamo.config, "fail_on_recompile_limit_hit"):
                # Also fail instead of silently falling back to eager after too many recompiles.
                torch._dynamo.config.fail_on_recompile_limit_hit = True
            raw_step = type(opt).step
            step_fns.append(torch.compile(lambda o=opt: raw_step(o), fullgraph=True))
        else:
            step_fns.append(opt.step)

    muon_params, adamw_params = split_params(model)
    n_muon = sum(p.numel() for p in muon_params)
    n_adamw = sum(p.numel() for p in adamw_params)
    logger.info(f"Parameters: {n_muon:,} in Muon-eligible matrices, {n_adamw:,} other "
                f"({100 * n_muon / (n_muon + n_adamw):.1f}% Muon-eligible)")

    import wandb
    run = wandb.init(project="muonmol-ref", name=args.run_name, mode=args.wandb_mode,
                     dir=work_dir, config={**vars(args), "n_muon_params": n_muon,
                                           "n_other_params": n_adamw,
                                           "torch": torch.__version__})

    train_csv = open(os.path.join(work_dir, "train_loss.csv"), "w", newline="")
    train_writer = csv.writer(train_csv)
    train_writer.writerow(["step", "train_bits_per_byte", "lr", "opt_step_ms"])
    val_csv = open(os.path.join(work_dir, "val_loss.csv"), "w", newline="")
    val_writer = csv.writer(val_csv)
    val_writer.writerow(["step", "val_bits_per_byte"])

    def log_val(step):
        val = evaluate(model, val_loader, device)
        val_writer.writerow([step, val])
        val_csv.flush()
        run.log({"val_bits_per_byte": val}, step=step)
        logger.info(f"Step {step}: validation {val:.4f} bits/byte")

    log_val(0)
    n_steps = args.stop_after or args.max_steps
    opt_times, lr_warned = [], False
    t_start = time.perf_counter()
    step, data_iter = 0, iter(train_loader)

    while step < n_steps:
        try:
            batch = next(data_iter)
        except StopIteration:  # repeat the data if it runs out (logged above)
            data_iter = iter(train_loader)
            batch = next(data_iter)
        batch = batch.to(device)

        loss = model(input_ids=batch, labels=batch).loss
        loss.backward()

        sync(device)
        t0 = time.perf_counter()
        for fn in step_fns:
            fn()
        sync(device)
        opt_ms = (time.perf_counter() - t0) * 1000
        opt_times.append(opt_ms)

        for sched in schedulers:
            sched.step()
        for _, opt in optimizers:
            opt.zero_grad(set_to_none=True)

        # With --compile-opt the LR must stay a tensor, otherwise every LR change recompiles.
        if args.compile_opt and not lr_warned:
            if not isinstance(optimizers[0][1].param_groups[0]["lr"], torch.Tensor):
                logger.warning("LR is no longer a tensor after the scheduler step: "
                               "torch.compile may recompile every step")
                lr_warned = True

        step += 1
        lr = float(optimizers[0][1].param_groups[0]["lr"])
        train_bits = loss.item() / math.log(2)  # model loss is in nats; bits like validation
        train_writer.writerow([step, train_bits, lr, opt_ms])
        run.log({"train_bits_per_byte": train_bits, "lr": lr, "opt_step_ms": opt_ms,
                 "bytes_seen": step * args.batch_size * args.seq_len}, step=step)
        if step % 10 == 0:
            train_csv.flush()
            logger.info(f"Step {step}/{n_steps} LR {lr:.2e} train {train_bits:.4f} bits/byte "
                        f"opt step {opt_ms:.1f} ms")
        if step % args.eval_every == 0 or step == n_steps:
            log_val(step)

    total_min = (time.perf_counter() - t_start) / 60
    steady = opt_times[10:] if len(opt_times) > 20 else opt_times
    summary = {
        "total_minutes": total_min,
        "first_opt_step_ms": opt_times[0],  # includes compile time with --compile-opt
        "median_opt_step_ms": float(np.median(steady)),
        "minutes_per_100_steps": total_min / step * 100,
    }
    if args.compile_opt:
        try:
            from torch._dynamo.utils import counters
            stats, breaks = dict(counters["stats"]), dict(counters["graph_break"])
            summary["dynamo_stats"] = str(stats)
            summary["dynamo_graph_breaks"] = str(breaks)
            # Expected: no graph breaks, and few graphs (1 = compiled once, never recompiled).
            if breaks:
                logger.error("COMPILE CHECK FAILED: graph breaks occurred, see dynamo_graph_breaks")
            if stats.get("unique_graphs", 0) > 2:
                logger.warning(f"Compiled {stats['unique_graphs']} different graphs: "
                               "the Muon step was recompiled, timing may include compile time")
        except Exception as e:  # internal API, may change between versions
            summary["dynamo_stats"] = f"not available ({e})"
    for k, v in summary.items():
        run.summary[k] = v
        logger.info(f"{k}: {v}")

    train_csv.close()
    val_csv.close()
    run.finish()


if __name__ == "__main__":
    main()
