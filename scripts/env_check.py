"""ENV stage: environment check for the Muon vs AdamW project (see PLAN.md, stage ENV).

Run it inside the environment you will train in (WSL2 + PyTorch xpu build):

    python scripts/env_check.py                              # all checks, wandb offline
    python scripts/env_check.py --report env_check_report.md # also save a Markdown report
    python scripts/env_check.py --wandb-mode online          # also test wandb login + upload
    python scripts/env_check.py --device cpu                 # dry run of the script itself on CPU
    python scripts/env_check.py --no-compile                 # skip the (slow) torch.compile check

Checks (required = must work on the A770 or have a documented CPU fallback):

    | Check                                  | Required |
    |----------------------------------------|----------|
    | System info (Python, PyTorch, OS, WSL) | info     |
    | xpu device + fp32 ops vs CPU           | yes      |
    | bf16 matmul + autocast forward/backward| yes      |
    | Matmul throughput fp32 vs bf16         | info     |
    | fp64 on the device                     | info     |
    | Attention (SDPA, causal, bf16)         | yes      |
    | torch.optim.Muon + AdamW steps         | yes      |
    | torch.compile (tiny transformer block) | info (eager mode is the fallback) |
    | torch.linalg.svdvals                   | yes (CPU fallback allowed) |
    | wandb init / log / finish              | yes      |

Done when (PLAN.md): every op runs on the A770, or a CPU fallback is documented.
Status FALLBACK = works, but not on the GPU as intended (e.g. PyTorch silently ran an
op on the CPU, or torch.compile is unusable and eager mode must be used).
Exit code: 0 if no required check failed, 1 otherwise.

The tolerances below are loose sanity bounds chosen by us. They are meant to catch
broken kernels, not to benchmark precision. The measured values are always printed.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import inspect
import math
import os
import platform
import sys
import tempfile
import time
import traceback
import warnings
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

FP32_TOL = 1e-3   # relative error, device fp32 matmul vs CPU fp32
BF16_TOL = 2e-2   # relative error, bf16 matmul vs fp32 (bf16 keeps ~3 significant digits)
ATTN_TOL = 5e-2   # relative error, bf16 attention (output and gradients) vs fp32 reference
SVD_TOL = 1e-3    # max |s_device - s_cpu| / max(s_cpu)
COMPILE_TOL = 5e-2  # relative error, compiled vs eager (output and gradients, bf16 autocast)
MUON_LR = 0.02    # learning rate used only to measure the size of one Muon step

PASS, FAIL, FALLBACK, INFO, SKIP = "PASS", "FAIL", "FALLBACK", "INFO", "SKIP"

DEVICE = torch.device("xpu")  # overwritten by --device in main()
CPU = torch.device("cpu")


@dataclass
class Result:
    name: str
    required: bool
    status: str
    details: list[str] = field(default_factory=list)


RESULTS: list[Result] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sync(device: torch.device) -> None:
    """Wait until all queued GPU work is finished (needed for correct timing)."""
    if device.type == "xpu":
        torch.xpu.synchronize()


def time_ms(fn, device: torch.device, warmup: int = 3, repeats: int = 10) -> float:
    """Median wall-clock time of fn() in milliseconds."""
    for _ in range(warmup):
        fn()
    sync(device)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative error ||a - b|| / ||b||, computed on CPU in float32."""
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def first_line(e: BaseException, limit: int = 250) -> str:
    text = str(e).strip().splitlines()
    return f"{type(e).__name__}: {text[0][:limit] if text else ''}"


def run_check(name: str, required: bool, fn) -> Result:
    """Run one check; catch errors and warnings; print and store the result."""
    details: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            status = fn(details)
        except Exception as e:  # a crash in a required check is a FAIL
            status = FAIL if required else INFO
            details.append(f"error: {first_line(e)}")
            traceback.print_exc(limit=4)
    # Warnings can reveal silent CPU fallbacks or slow paths, so we keep them.
    seen = set()
    cpu_fallback = False
    for w in caught:
        full = str(w.message)
        if "fallback" in full.lower() and "xpu" in full.lower() and "cpu" in full.lower():
            cpu_fallback = True
        msg = full.strip().splitlines()[0][:200] if full.strip() else ""
        if msg and msg not in seen:
            seen.add(msg)
            details.append(f"warning: {msg}")
    # A PASS with a silent XPU -> CPU fallback is not a real GPU pass.
    if cpu_fallback and status == PASS:
        status = FALLBACK
        details.append("-> PyTorch ran at least one op of this check on the CPU (see warning). "
                       "Set PYTORCH_DEBUG_XPU_FALLBACK=1 to see which op.")
    result = Result(name, required, status, details)
    RESULTS.append(result)
    print(f"[{status:8}] {name}")
    for d in details:
        print(f"             {d}")
    print()
    return result


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_system(details: list[str]) -> str:
    details.append(f"Python {platform.python_version()}, PyTorch {torch.__version__}")
    details.append(f"Platform: {platform.platform()}")

    distro = "unknown (/etc/os-release not found)"
    try:
        with open("/etc/os-release", encoding="utf-8") as f:
            info = dict(line.rstrip("\n").split("=", 1) for line in f if "=" in line)
        distro = info.get("PRETTY_NAME", "unknown").strip('"')
    except OSError:
        pass
    details.append(f"Distribution: {distro}")

    wsl = "unknown (/proc/version not found)"
    try:
        with open("/proc/version", encoding="utf-8") as f:
            wsl = str("microsoft" in f.read().lower())
    except OSError:
        pass
    details.append(f"Running in WSL: {wsl}")

    # Environment variables that can change GPU behaviour (e.g. enable CPU fallbacks).
    for key in sorted(os.environ):
        if key.startswith(("PYTORCH_", "SYCL_", "ZE_", "ONEAPI_", "UR_")):
            details.append(f"env {key}={os.environ[key]}")
        elif key.startswith("WANDB_"):
            details.append(f"env {key} is set (value hidden)")
    return INFO


def check_device(details: list[str]) -> str:
    if DEVICE.type == "cpu":
        details.append("--device cpu: dry run of the script, the GPU is NOT tested")
        return SKIP
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        details.append("torch.xpu.is_available() is False")
        details.append("Check: Intel GPU driver on Windows, and that the xpu build of PyTorch "
                       "is installed (LITERATURE.md -> 'Getting Started on Intel GPU')")
        return FAIL

    details.append(f"Device count: {torch.xpu.device_count()}")
    details.append(f"Device 0: {torch.xpu.get_device_name(0)}")
    props = torch.xpu.get_device_properties(0)
    total = getattr(props, "total_memory", None)
    if total:
        details.append(f"Total memory: {total / 2**30:.1f} GiB")
    details.append(f"Properties: {props}")

    torch.manual_seed(0)
    a, b = torch.randn(1024, 1024), torch.randn(1024, 1024)
    err = rel_err((a.to(DEVICE) @ b.to(DEVICE)).cpu(), a @ b)
    details.append(f"fp32 matmul rel. error vs CPU: {err:.1e} (bound {FP32_TOL:.0e})")
    return PASS if err < FP32_TOL else FAIL


def check_bf16(details: list[str]) -> str:
    if DEVICE.type == "xpu" and hasattr(torch.xpu, "is_bf16_supported"):
        details.append(f"torch.xpu.is_bf16_supported(): {torch.xpu.is_bf16_supported()}")

    # 1) Plain bf16 matmul vs fp32 on the same device.
    torch.manual_seed(0)
    a = torch.randn(1024, 1024, device=DEVICE)
    b = torch.randn(1024, 1024, device=DEVICE)
    err = rel_err(a.bfloat16() @ b.bfloat16(), a @ b)
    details.append(f"bf16 matmul rel. error vs fp32: {err:.1e} (bound {BF16_TOL:.0e})")

    # 2) Autocast forward + backward through a small MLP (how training will use bf16).
    model = torch.nn.Sequential(
        torch.nn.Linear(256, 1024), torch.nn.GELU(), torch.nn.Linear(1024, 256)
    ).to(DEVICE)
    x = torch.randn(64, 256, device=DEVICE)
    with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
        y = model(x)
        loss = y.float().pow(2).mean()
    loss.backward()
    grads_ok = all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    details.append(f"autocast output dtype: {y.dtype}, loss finite: {bool(torch.isfinite(loss))}, "
                   f"all grads finite: {grads_ok}")

    ok = err < BF16_TOL and y.dtype == torch.bfloat16 and grads_ok and bool(torch.isfinite(loss))
    return PASS if ok else FAIL


def check_throughput(details: list[str]) -> str:
    n = 4096 if DEVICE.type == "xpu" else 1024
    flops = 2 * n**3
    for dtype in (torch.float32, torch.bfloat16):
        a = torch.randn(n, n, device=DEVICE, dtype=dtype)
        b = torch.randn(n, n, device=DEVICE, dtype=dtype)
        ms = time_ms(lambda: a @ b, DEVICE)
        details.append(f"{str(dtype).replace('torch.', ''):9} {n}x{n} matmul: {ms:7.2f} ms "
                       f"= {flops / (ms / 1000) / 1e12:6.2f} TFLOP/s")
    details.append("Useful later for BUDGET (time per run); not a pass/fail check")
    return INFO


def check_fp64(details: list[str]) -> str:
    if DEVICE.type == "xpu":
        props = torch.xpu.get_device_properties(0)
        if hasattr(props, "has_fp64"):
            details.append(f"Device reports has_fp64: {props.has_fp64}")
    try:
        x = torch.ones(8, dtype=torch.float64, device=DEVICE)
        details.append(f"fp64 tensor op on device works (result {(x * 2).sum().item()})")
    except Exception as e:
        details.append(f"fp64 on device fails: {first_line(e)}")
        details.append("-> keep fp64 tensors (e.g. from numpy) on CPU; cast to float32 before .to(device)")
    return INFO


def check_attention(details: list[str]) -> str:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    B, H, T, D = 32, 8, 128, 64  # batch, heads, sequence length, head dim
    details.append(f"Shape (B, H, T, D) = {(B, H, T, D)}, causal")
    torch.manual_seed(0)
    q, k, v, g = (torch.randn(B, H, T, D, device=DEVICE) for _ in range(4))

    def fwd_bwd(dtype, backend=None):
        qq, kk, vv = (t.detach().to(dtype).requires_grad_(True) for t in (q, k, v))
        ctx = sdpa_kernel(backend) if backend is not None else contextlib.nullcontext()
        with ctx:
            out = F.scaled_dot_product_attention(qq, kk, vv, is_causal=True)
        out.backward(g.to(dtype))
        return out, qq.grad, kk.grad, vv.grad

    # Reference: plain math implementation in fp32.
    ref = fwd_bwd(torch.float32, SDPBackend.MATH)

    # 1) Default dispatch in bf16 (what the model will use).
    out = fwd_bwd(torch.bfloat16)
    errs = [rel_err(o, r) for o, r in zip(out, ref)]
    details.append("default bf16 rel. errors vs fp32 math (out, dq, dk, dv): "
                   + ", ".join(f"{e:.1e}" for e in errs) + f" (bound {ATTN_TOL:.0e})")
    default_ms = time_ms(lambda: fwd_bwd(torch.bfloat16), DEVICE, warmup=2, repeats=5)
    details.append(f"default bf16 fwd+bwd: {default_ms:.2f} ms")

    # 2) Which backends run on this device? (CUDNN_ATTENTION is CUDA-only, so skipped.)
    for name in ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "OVERRIDEABLE", "MATH"):
        backend = getattr(SDPBackend, name, None)
        if backend is None:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = fwd_bwd(torch.bfloat16, backend)
                ms = time_ms(lambda: fwd_bwd(torch.bfloat16, backend), DEVICE, warmup=2, repeats=5)
            details.append(f"backend {name:19}: works, fwd+bwd {ms:7.2f} ms, "
                           f"out rel. error {rel_err(res[0], ref[0]):.1e}")
        except Exception as e:
            details.append(f"backend {name:19}: not available ({first_line(e, 120)})")
    details.append("If default is much faster than MATH, a fused kernel is in use")

    return PASS if all(e < ATTN_TOL for e in errs) else FAIL


def check_optimizers(details: list[str]) -> str:
    ok = True

    # --- Muon -------------------------------------------------------------
    if not hasattr(torch.optim, "Muon"):
        details.append(f"torch.optim.Muon not found in PyTorch {torch.__version__}")
        ok = False
    else:
        details.append(f"torch.optim.Muon{inspect.signature(torch.optim.Muon)}")
        # Expected update RMS / lr if Newton-Schulz orthogonalized exactly (my calc):
        #   plain Muon (LITERATURE.md, Liu et al.):  sqrt(1 / max(A, B))
        #   "original"        (torch docs): plain * sqrt(max(1, A / B))
        #   "match_rms_adamw" (torch docs): plain * 0.2 * sqrt(max(A, B)) = 0.2
        # Newton-Schulz is approximate, so measured values are usually a bit lower.
        for variant in ("original", "match_rms_adamw"):
            for A, B in ((512, 512), (1024, 256)):
                torch.manual_seed(0)
                w = torch.nn.Parameter(0.02 * torch.randn(A, B, device=DEVICE))
                opt = torch.optim.Muon([w], lr=MUON_LR, weight_decay=0.0, adjust_lr_fn=variant)
                w0 = w.detach().clone()
                w.grad = torch.randn_like(w)
                opt.step()
                finite = bool(torch.isfinite(w).all())
                rms = (w.detach() - w0).float().pow(2).mean().sqrt().item() / MUON_LR
                plain = math.sqrt(1 / max(A, B))
                ideal = plain * math.sqrt(max(1, A / B)) if variant == "original" else 0.2
                details.append(f"Muon {variant:15} {A:4}x{B:<4}: finite {finite}, "
                               f"update RMS / lr = {rms:.3f} (ideal {ideal:.3f})")
                ok = ok and finite and rms > 0

    # --- AdamW (baseline) ---------------------------------------------------
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(256, 256, device=DEVICE))
    opt = torch.optim.AdamW([w], lr=1e-3)
    w.grad = torch.randn_like(w)
    opt.step()
    finite = bool(torch.isfinite(w).all())
    details.append(f"AdamW (default implementation): step finite {finite}")
    ok = ok and finite

    try:  # fused AdamW is only a speed option, so it does not affect the status
        w2 = torch.nn.Parameter(torch.randn(256, 256, device=DEVICE))
        opt2 = torch.optim.AdamW([w2], lr=1e-3, fused=True)
        w2.grad = torch.randn_like(w2)
        opt2.step()
        details.append("AdamW fused=True: works (optional)")
    except Exception as e:
        details.append(f"AdamW fused=True: not available, use default ({first_line(e, 120)})")

    return PASS if ok else FAIL


class TinyBlock(torch.nn.Module):
    """One Llama-like decoder block (simplified: GELU MLP instead of SwiGLU).

    Only used to test torch.compile on the ops our model will use:
    RMSNorm, linear layers, causal SDPA attention, residuals.
    """

    def __init__(self, d: int = 256, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.norm1 = torch.nn.RMSNorm(d)
        self.qkv = torch.nn.Linear(d, 3 * d, bias=False)
        self.proj = torch.nn.Linear(d, d, bias=False)
        self.norm2 = torch.nn.RMSNorm(d)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d, 4 * d, bias=False), torch.nn.GELU(), torch.nn.Linear(4 * d, d, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(self.norm1(x)).view(B, T, 3, self.heads, D // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each (B, heads, T, head_dim)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.norm2(x))


def check_compile(details: list[str], enabled: bool = True) -> str:
    if not enabled:
        details.append("--no-compile: skipped")
        return SKIP
    import torch._dynamo

    try:
        import triton
        details.append(f"triton {triton.__version__} (Inductor uses it to generate GPU kernels)")
    except ImportError:
        details.append("triton is not importable (Inductor needs it for GPU kernels)")

    torch.manual_seed(0)
    model = TinyBlock().to(DEVICE)
    x = torch.randn(32, 64, 256, device=DEVICE)  # batch 32, 64 tokens (SMILES are short), d = 256
    g = torch.randn(32, 64, 256, device=DEVICE)  # fixed upstream gradient

    def train_step(m):
        """Forward + backward in bf16 autocast; returns output and gradients."""
        m.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
            y = m(x)
        (y.float() * g).sum().backward()
        return y.detach().float(), [p.grad.detach().float().clone() for p in m.parameters()]

    y_eager, grads_eager = train_step(model)

    # fullgraph=True: fail loudly on graph breaks instead of silently running parts in eager mode.
    torch._dynamo.reset()
    compiled = torch.compile(model, fullgraph=True)
    try:
        t0 = time.perf_counter()
        y_comp, grads_comp = train_step(compiled)  # first call compiles forward and backward
        sync(DEVICE)
        compile_s = time.perf_counter() - t0
    except Exception as e:
        details.append(f"torch.compile fails: {first_line(e, 300)}")
        details.append("-> use eager mode for now; REF re-tests compile on the toy model")
        return FALLBACK
    details.append(f"first compiled call (compile forward + backward): {compile_s:.1f} s")

    err_out = rel_err(y_comp, y_eager)
    err_grad = max(rel_err(gc, ge) for gc, ge in zip(grads_comp, grads_eager))
    details.append(f"compiled vs eager rel. error: output {err_out:.1e}, worst gradient {err_grad:.1e} "
                   f"(bound {COMPILE_TOL:.0e})")

    eager_ms = time_ms(lambda: train_step(model), DEVICE)
    comp_ms = time_ms(lambda: train_step(compiled), DEVICE)
    details.append(f"fwd+bwd: eager {eager_ms:.2f} ms, compiled {comp_ms:.2f} ms "
                   f"(speedup {eager_ms / comp_ms:.2f}x)")

    if err_out < COMPILE_TOL and err_grad < COMPILE_TOL:
        return PASS
    details.append("-> compiled results differ from eager: use eager mode until REF checks it")
    return FALLBACK


def check_svdvals(details: list[str]) -> str:
    # Weights must be float32 for svdvals (bf16 is generally not supported), so we test float32.
    torch.manual_seed(0)
    m = torch.randn(256, 1024)
    ref = torch.linalg.svdvals(m)
    cpu_ms = time_ms(lambda: torch.linalg.svdvals(m), CPU, warmup=1, repeats=5)
    details.append(f"CPU: works, {cpu_ms:.1f} ms for a 256x1024 float32 matrix")
    if DEVICE.type == "cpu":
        return PASS

    try:
        md = m.to(DEVICE)
        s = torch.linalg.svdvals(md)
        diff = ((s.cpu() - ref).abs().max() / ref.max()).item()
        dev_ms = time_ms(lambda: torch.linalg.svdvals(md), DEVICE, warmup=1, repeats=5)
        details.append(f"{DEVICE.type}: works, {dev_ms:.1f} ms, max rel. diff vs CPU {diff:.1e} "
                       f"(bound {SVD_TOL:.0e})")
        if diff < SVD_TOL:
            return PASS
        details.append("-> device result differs from CPU: use CPU fallback (weight.float().cpu())")
        return FALLBACK
    except Exception as e:
        details.append(f"{DEVICE.type}: fails ({first_line(e, 150)})")
        details.append("-> CPU fallback: torch.linalg.svdvals(weight.detach().float().cpu())")
        return FALLBACK


def check_wandb(details: list[str], mode: str) -> str:
    if mode == "disabled":
        details.append("--wandb-mode disabled: skipped")
        return SKIP
    try:
        import wandb
    except ImportError:
        details.append("wandb is not installed (pip install wandb)")
        return FAIL

    details.append(f"wandb {wandb.__version__}, mode {mode}")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        run = wandb.init(project="muonmol-env-check", mode=mode, dir=tmp,
                         config={"torch": torch.__version__, "device": str(DEVICE)})
        for step in range(5):
            run.log({"dummy_loss": 1.0 / (step + 1)}, step=step)
        try:
            url = run.url
        except Exception:
            url = None
        run.finish()
    if mode == "online" and url:
        details.append(f"run URL: {url}")
    details.append("init / log / finish worked")
    return PASS


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(path: str) -> None:
    lines = [
        "# ENV check report",
        "",
        f"- Date: {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"- Command: `{' '.join(sys.argv)}`",
        f"- PyTorch {torch.__version__}, device `{DEVICE}`",
        "",
        "| Check | Required | Status |",
        "|---|---|---|",
    ]
    for r in RESULTS:
        lines.append(f"| {r.name} | {'yes' if r.required else 'info'} | {r.status} |")
    lines.append("")
    for r in RESULTS:
        lines.append(f"## {r.name}: {r.status}")
        lines.append("")
        lines.extend(f"- {d}" for d in r.details)
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report written to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    global DEVICE
    parser = argparse.ArgumentParser(description="ENV stage: environment check")
    parser.add_argument("--device", default="xpu", choices=["xpu", "cpu"],
                        help="device to test (cpu = dry run of this script)")
    parser.add_argument("--wandb-mode", default="offline", choices=["offline", "online", "disabled"],
                        help="online also tests login and upload (run 'wandb login' first)")
    parser.add_argument("--report", default=None, help="optional path for a Markdown report")
    parser.add_argument("--no-compile", action="store_true", help="skip the torch.compile check")
    args = parser.parse_args()
    DEVICE = torch.device(args.device)

    print(f"ENV check on device '{DEVICE}'\n")
    run_check("System info", False, check_system)
    device_result = run_check("Device + fp32 ops", True, check_device)

    if device_result.status == FAIL:
        print("Device not usable, skipping the GPU checks.\n")
    else:
        run_check("bf16 (matmul + autocast fwd/bwd)", True, check_bf16)
        run_check("Matmul throughput", False, check_throughput)
        run_check("fp64 on device", False, check_fp64)
        run_check("Attention (SDPA, causal, bf16)", True, check_attention)
        run_check("Optimizers (torch.optim.Muon, AdamW)", True, check_optimizers)
        run_check("torch.compile (tiny transformer block)", False,
                  lambda d: check_compile(d, enabled=not args.no_compile))
        run_check("SVD (torch.linalg.svdvals)", True, check_svdvals)
    run_check("wandb", True, lambda d: check_wandb(d, args.wandb_mode))

    print("Summary")
    print("-------")
    for r in RESULTS:
        print(f"  {r.status:8}  {'required' if r.required else 'info    '}  {r.name}")

    failed = [r.name for r in RESULTS if r.required and r.status == FAIL]
    fallbacks = [r.name for r in RESULTS if r.status == FALLBACK]
    print()
    if failed:
        print(f"ENV NOT done: failed required checks: {', '.join(failed)}")
    elif DEVICE.type == "cpu":
        print("Dry run on CPU finished. Run again with --device xpu to test the GPU.")
    else:
        print("ENV done-when criterion met" + (f" (fallback for: {', '.join(fallbacks)})"
                                                if fallbacks else ""))
    if args.report:
        write_report(args.report)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
