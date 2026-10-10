# ENV check report

- Date: 2026-10-10T18:36:38
- Command: `/mnt/c/Users/mbrun/Documents/University/Master Thesis/Muon on Molucules Generation/scripts/env_check.py --report env_check_report.md --wandb-mode online`
- PyTorch 2.12.1+xpu, device `xpu`

| Check | Required | Status |
|---|---|---|
| System info | info | INFO |
| Device + fp32 ops | yes | PASS |
| bf16 (matmul + autocast fwd/bwd) | yes | PASS |
| Matmul throughput | info | INFO |
| fp64 on device | info | INFO |
| Attention (SDPA, causal, bf16) | yes | PASS |
| Optimizers (torch.optim.Muon, AdamW) | yes | PASS |
| SVD (torch.linalg.svdvals) | yes | PASS |
| wandb | yes | PASS |

## System info: INFO

- Python 3.11.15, PyTorch 2.12.1+xpu
- Platform: Linux-6.18.33.1-microsoft-standard-WSL2-x86_64-with-glibc2.43
- Distribution: Ubuntu 26.04 LTS
- Running in WSL: True

## Device + fp32 ops: PASS

- Device count: 1
- Device 0: Intel(R) Graphics [0x56a0]
- Total memory: 15.6 GiB
- Properties: _XpuDeviceProperties(name='Intel(R) Graphics [0x56a0]', platform_name='Intel(R) oneAPI Unified Runtime over Level-Zero', type='gpu', device_id=0x56A0, uuid=8680a056-0800-0000-0c00-000000000000, driver_version='1.14.37020', total_memory=15932MB, local_mem_size=64KB, max_compute_units=512, memory_clock_rate=0MHz, memory_bus_width=64-bit, gpu_eu_count=512, gpu_subslice_count=64, max_work_group_size=1024, max_num_sub_groups=128, sub_group_sizes=[8 16 32], has_fp16=1, has_fp64=0, has_atomic64=1)
- fp32 matmul rel. error vs CPU: 6.0e-07 (bound 1e-03)

## bf16 (matmul + autocast fwd/bwd): PASS

- torch.xpu.is_bf16_supported(): True
- bf16 matmul rel. error vs fp32: 2.9e-03 (bound 2e-02)
- autocast output dtype: torch.bfloat16, loss finite: True, all grads finite: True

## Matmul throughput: INFO

- float32   4096x4096 matmul:    9.11 ms =  15.08 TFLOP/s
- bfloat16  4096x4096 matmul:    2.30 ms =  59.66 TFLOP/s
- Useful later for BUDGET (time per run); not a pass/fail check

## fp64 on device: INFO

- Device reports has_fp64: False
- fp64 on device fails: RuntimeError: Required aspect fp64 is not supported on the device
- -> keep fp64 tensors (e.g. from numpy) on CPU; cast to float32 before .to(device)

## Attention (SDPA, causal, bf16): PASS

- Shape (B, H, T, D) = (32, 8, 128, 64), causal
- default bf16 rel. errors vs fp32 math (out, dq, dk, dv): 3.0e-03, 3.9e-03, 3.9e-03, 3.0e-03 (bound 5e-02)
- default bf16 fwd+bwd: 3.33 ms
- backend FLASH_ATTENTION    : not available (RuntimeError: No available kernel. Aborting execution.)
- backend EFFICIENT_ATTENTION: works, fwd+bwd    3.29 ms, out rel. error 3.0e-03
- backend OVERRIDEABLE       : not available (RuntimeError: No available kernel. Aborting execution.)
- backend MATH               : works, fwd+bwd    3.42 ms, out rel. error 3.0e-03
- If default is much faster than MATH, a fused kernel is in use

## Optimizers (torch.optim.Muon, AdamW): PASS

- torch.optim.Muon(params: collections.abc.Iterable[torch.Tensor] | collections.abc.Iterable[dict[str, typing.Any]] | collections.abc.Iterable[tuple[str, torch.Tensor]], lr: float = 0.001, weight_decay: float = 0.1, momentum: float = 0.95, nesterov: bool = True, ns_coefficients: tuple[float, float, float] = (3.4445, -4.775, 2.0315), eps: float = 1e-07, ns_steps: int = 5, adjust_lr_fn: str | None = None) -> None
- Muon original         512x512 : finite True, update RMS / lr = 0.041 (ideal 0.044)
- Muon original        1024x256 : finite True, update RMS / lr = 0.060 (ideal 0.062)
- Muon match_rms_adamw  512x512 : finite True, update RMS / lr = 0.188 (ideal 0.200)
- Muon match_rms_adamw 1024x256 : finite True, update RMS / lr = 0.191 (ideal 0.200)
- AdamW (default implementation): step finite True
- AdamW fused=True: not available, use default (RuntimeError: Required aspect fp64 is not supported on the device)

## SVD (torch.linalg.svdvals): PASS

- CPU: works, 14.4 ms for a 256x1024 float32 matrix
- xpu: works, 18.5 ms, max rel. diff vs CPU 0.0e+00 (bound 1e-03)
- warning: Aten Op fallback from XPU to CPU happends. This may have performance implications. If need debug the fallback ops please set environment variable `PYTORCH_DEBUG_XPU_FALLBACK=1`  (Triggered internally 

## wandb: PASS

- wandb 0.30.0, mode online
- run URL: https://wandb.ai/brunner2-johannes-kepler-universit-t-linz/muonmol-env-check/runs/x999vtbd
- init / log / finish worked
