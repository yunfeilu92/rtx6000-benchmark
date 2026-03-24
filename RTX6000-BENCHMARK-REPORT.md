# RTX PRO 6000 Blackwell Benchmark Report

## Overview

PLUTO (autonomous driving planning model) training benchmark on AWS g7e.48xlarge with 8x NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs.

Adapted from [h200-benchmark](https://github.com/yunfeilu92/h200-benchmark) for Blackwell architecture comparison.

## Infrastructure

| Component | Spec |
|---|---|
| Instance | g7e.48xlarge |
| GPU | 8x NVIDIA RTX PRO 6000 Blackwell Server Edition |
| GPU Memory | 96 GB GDDR7 per card (768 GB total) |
| GPU Architecture | Blackwell, SM 12.0, 24,064 CUDA Cores |
| GPU TDP | 600W per card |
| vCPU | 192 |
| RAM | 2,048 GiB |
| Local NVMe | 14 TB |
| Network | PCIe 5.0 x16 (no NVLink) |
| Price | ~$33.14/hr On-Demand |

## RTX PRO 6000 Compute Specs

| Precision | TFLOPS | With Sparsity |
|---|---|---|
| FP64 | 1.97 | - |
| FP32 | 125 | - |
| TF32 Tensor | 126 | 252 |
| FP16/BF16 Tensor | 252 | 504 |
| FP8 Tensor | 504 | 1,008 |
| FP4 Tensor | 2,015 | 4,030 |
| Memory Bandwidth | 1,600 GB/s | - |

## Software Stack

| Component | Version | Notes |
|---|---|---|
| Driver | 580.126.09 | CUDA 13.0 |
| Python | 3.12.13 | via Miniconda |
| PyTorch | **2.10.0+cu128** | MUST be >= 2.10.0 for NCCL fix |
| NCCL | **2.27.5** | 2.26.2 (PyTorch 2.7) hangs on Blackwell |
| cuBLAS | **12.9.1.4** | 12.8.x crashes batched GEMM on Blackwell |
| NATTEN | 0.21.5 | API changed from 0.14.x |
| PyTorch Lightning | 2.0.1 | |
| numpy | 1.26.4 | < 2.0 for PL compat |
| torchmetrics | 0.9.3 | < 0.10 for compute_on_step |

## Benchmark Configuration

### Algorithm: PLUTO
- Imitation learning for autonomous driving path planning
- Model size: **4.1M parameters** (~16MB)
- Dataset: nuPlan v1.1 (8,733 scenarios, 775K training samples)

### Final Training Config
| Parameter | Value |
|---|---|
| Batch Size | 256 (32 per GPU) |
| GPUs | 8 (DDP) |
| Epochs | 25 |
| Learning Rate | 1e-3 (Cosine + 3 epoch warmup) |
| Precision | FP32 |
| FPN Upsample | nearest (optimized from linear) |
| Strategy | DDP (find_unused_parameters=false) |
| DataLoader Workers | 20 |
| pin_memory | True |

## Results

### Feature Cache (Step 1)
| Metric | Value |
|---|---|
| Duration | **2h 33m** |
| Scenarios | 25,000 |
| Ray Workers | 40 |
| Cache Size | ~460 GB |

### Training (Step 2) — Optimization Journey

#### Original: FPN linear, FP32, batch=64
| Metric | Value |
|---|---|
| Epoch Time | **~22 min** |
| GPU Utilization | 96-100% (misleading — GPU busy-waiting on slow linear interpolate) |
| VRAM Usage | ~25 GB / 95 GB (26%) |
| Power Draw | ~135W / 600W (22% TDP) |

#### After batch=256, FP32, linear
| Metric | Value |
|---|---|
| Epoch Time | **~28 min** |
| GPU Utilization | 100% |
| VRAM Usage | ~65-91 GB / 95 GB (68-95%) |
| Power Draw | ~100W / 600W (17% TDP) |

#### Final: FPN nearest, FP32, batch=256 (current run, in progress)
| Metric | Value |
|---|---|
| Epoch Time | **~8 min** |
| GPU Utilization | avg 62%, bursts 0-100% (real compute cycles) |
| VRAM Usage | ~62-94 GB / 95 GB (65-98%) |
| Power Draw | ~248W / 600W (41% TDP) |
| Temperature | 42-43°C |

### Loss Convergence (FPN nearest, 25 epochs, in progress)

| Epoch | Val Loss | Trend |
|---|---|---|
| 3 | 4.6185 | |
| 4 | 3.7465 | ↓ |
| 5 | 3.5641 | ↓ |
| 6 | 3.4671 | ↓ |
| 7 | 3.2991 | ↓ |

Loss is steadily converging. Training is ongoing — will update with final results.

### Optimization Impact Summary

| Configuration | Epoch Time | Power (TDP%) | Speedup |
|---|---|---|---|
| linear + FP32 + batch=64 | 22 min | 135W (22%) | 1x |
| linear + FP32 + batch=256 | 28 min | 100W (17%) | 0.8x |
| **nearest + FP32 + batch=256** | **8 min** | **248W (41%)** | **2.8x** |
| nearest + BF16 + batch=256 | 7.2 min | 255W (43%) | 3.1x |
| nearest + BF16 + batch=128 | 7.6 min | ~240W (40%) | 2.9x |
| nearest + BF16 + batch=512 | >30 min | ~97W (16%) | OOM-level slow |

## Profiling Analysis

### Before Optimization (FPN linear)

Single GPU, batch=256, FP32, 20 steps:

| Phase | Time per Step | Percentage |
|---|---|---|
| Data transfer (CPU->GPU) | 0.316s | 8.4% |
| **Forward pass** | **1.385s** | **36.9%** |
| Loss compute | 0.003s | 0.1% |
| **Backward pass** | **2.037s** | **54.3%** |
| Optimizer step | 0.008s | 0.2% |
| **Total** | **3.749s** | 100% |
| **Throughput** | **68 samples/sec** | |

### After Optimization (FPN nearest)

Single GPU, batch=256, FP32, 20 steps:

| Phase | Time per Step | Percentage |
|---|---|---|
| **Data transfer** | **0.284s** | **30.8%** |
| Forward pass | 0.221s | 24.0% |
| Loss compute | 0.003s | 0.3% |
| Backward pass | 0.408s | 44.3% |
| Optimizer step | 0.006s | 0.6% |
| **Total** | **0.921s** | 100% |
| **Throughput** | **278 samples/sec** | **4.1x faster** |

### Key Finding: FPN Upsample Was the #1 Bottleneck

`F.interpolate(mode="linear")` consumed the majority of GPU time:
- Linear interpolation is a pure memory-bandwidth operation — does NOT use Tensor Cores
- Forward: 1.385s → 0.221s (**6.3x faster**)
- Backward: 2.037s → 0.408s (**5.0x faster**)
- Same fix was discovered on H200 benchmark (89.4% GPU time on linear)

After fixing upsample, the bottleneck shifted to **CPU→GPU data transfer (31%)**, because nuPlan batches contain 50+ nested tensors that transfer individually.

### BF16 vs FP32 (after upsample fix)

| Metric | FP32 | BF16-mixed |
|---|---|---|
| Forward | 0.221s | 0.165s (**25% faster**) |
| Backward | 0.408s | 0.267s (**35% faster**) |
| Total/step | 0.921s | 0.767s (**17% faster**) |
| Throughput | 278 samp/sec | 334 samp/sec (**+20%**) |
| VRAM | ~65-91 GB | 41.6 GB (**55% less**) |

BF16 shows real speedup after the upsample fix because GPU compute is now a meaningful portion of step time. Before the fix, BF16 showed zero speedup (GPU was bottlenecked on linear interpolate which doesn't benefit from Tensor Cores).

### 8-GPU Batch Size Sweep (BF16, nearest)

| Batch | Epoch Time (3 epochs avg) | VRAM/GPU |
|---|---|---|
| 128 | **7.6 min** | ~32 GB (33%) |
| **256** | **7.2 min** | ~65 GB (68%) |
| 512 | >30 min (OOM-level) | ~96 GB (98%) |

Batch=256 is the sweet spot. Batch=512 causes memory pressure that kills throughput.

### GPU Utilization Deep Dive (8-GPU, nearest, batch=256, FP32)

30-second monitoring with 10 samples at 3-second intervals:

| Phase | GPU Util | Power | What's Happening |
|---|---|---|---|
| Compute burst | 85-100% | 230-280W | Forward + backward pass |
| DDP AllReduce | 0-7% | 220-240W | NCCL sync over PCIe (power still high) |
| Data transfer | 10-15% | 260-280W | CPU→GPU batch loading |

**Average: 62% GPU utilization, 248W power (41% TDP)**

The 4.1M parameter model cannot fully saturate 8x RTX PRO 6000 GPUs. Power draw at 41% TDP indicates significant headroom for larger models.

### Optimization Assessment

| Optimization | Tested? | Impact |
|---|---|---|
| **FPN upsample nearest** | Yes | **4x single-GPU, 2.8x 8-GPU** |
| pin_memory=True | Yes | Included in final config |
| BF16-mixed | Yes | 20% faster after upsample fix |
| torch.compile | Blocked | Triton segfaults on SM 12.0 ([#176426](https://github.com/pytorch/pytorch/issues/176426)) |
| Larger batch (512) | Yes | OOM-level slow |
| Gradient accumulation | Analyzed | Negative impact for small models |
| CUDA Graphs | Blocked | Dynamic shapes + SM 12.0 incompatible |
| Tensor packing in collate_fn | Not tested | Could reduce 31% data transfer overhead |
| CUDA stream double-buffering | Not tested | Could overlap transfer with compute |

## Blackwell Compatibility Issues (Critical)

### 1. NCCL 2.26.2 Completely Broken
PyTorch 2.7.0 ships NCCL 2.26.2 which **hangs indefinitely** on Blackwell during any collective operation.

**Fix**: Use PyTorch >= 2.10.0 (ships NCCL 2.27.5).

### 2. cuBLAS Batched GEMM Crash
cuBLAS 12.8.x crashes with `CUBLAS_STATUS_INVALID_VALUE` on `torch.bmm()` and all batched GEMM ops. Affects **all precisions**.

**Fix**: `pip install --force-reinstall --no-deps nvidia-cublas-cu12==12.9.1.4`

### 3. NCCL Socket Interface
g7e uses `enp135s0`, not `eth0`/`ens5`. Wrong setting causes "Bootstrap: no socket interface found".

**Fix**: `export NCCL_SOCKET_IFNAME=enp135s0`

### 4. Required NCCL Environment Variables
```bash
export NCCL_P2P_DISABLE=1     # P2P broken on Blackwell (NVIDIA/nccl#1999)
export NCCL_IB_DISABLE=1      # No InfiniBand on g7e
export NCCL_NVLS_ENABLE=0     # NVLink SHARP causes hangs
export NCCL_SOCKET_IFNAME=enp135s0
```

### 5. BF16 Dtype Mismatches in PLUTO
`torch.zeros()` without dtype + `nn.Embedding.weight` staying FP32 under autocast.

**Fix**: `patch_pluto_bf16.sh` (6 locations across 3 files)

### 6. PLUTO Code Patches Required
- Missing `__init__.py` in `src/` subdirectories (Python 3.12)
- NATTEN 0.21.x API change (`attn_drop` removed)
- LRScheduler `verbose` param removed in PyTorch 2.7+
- nuplan data directory symlinks needed

### 7. Triton / torch.compile Not Ready for SM 12.0
[PyTorch #176426](https://github.com/pytorch/pytorch/issues/176426): Triton segfaults on SM 12.0.

## GPU Comparison

### RTX PRO 6000 vs H200

| Metric | RTX PRO 6000 (g7e) | H200 (p5en) |
|---|---|---|
| Instance | g7e.48xlarge | p5en.48xlarge |
| GPU Count | 8 | 8 |
| GPU Memory | 96 GB GDDR7 | 141 GB HBM3e |
| Memory BW | 1,600 GB/s | 4,800 GB/s |
| FP32 TFLOPS | 125 | 67 |
| BF16 Tensor | 252 | ~1,979 |
| Interconnect | PCIe 5.0 | NVLink 900 GB/s |
| Price | ~$33/hr (OD) | ~$98/hr (CB) |

### RTX PRO 6000 vs L40S

| Metric | L40S (g6e) | RTX PRO 6000 (g7e) |
|---|---|---|
| Architecture | Ada Lovelace SM 8.9 | Blackwell SM 12.0 |
| FP32 | 91.6 TFLOPS | 125 TFLOPS (+36%) |
| BF16 Tensor | **362 TFLOPS** | 252 TFLOPS (L40S wins) |
| FP4 Tensor | N/A | 2,015 TFLOPS (exclusive) |
| VRAM | 48 GB GDDR6 | 96 GB GDDR7 (2x) |
| Memory BW | 864 GB/s | 1,600 GB/s (1.85x) |
| 8-GPU Price | $30.13/hr | $33.14/hr (+10%) |

### When to Use Which GPU

| Model Size | Recommendation |
|---|---|
| < 500M params (PLUTO-class) | **L40S** — cheaper, compute not bottleneck |
| 500M - 30B | **RTX PRO 6000** — 1.85x bandwidth, 2x VRAM |
| 30B+ inference | **RTX PRO 6000 required** — doesn't fit in L40S 48GB |
| FP4 quantized inference | **RTX PRO 6000 required** — L40S has no FP4 |
| HPC / large-scale training | **H200** — NVLink + HBM3e, different league |

## File Manifest

| File | Description |
|---|---|
| `setup.sh` | Full environment setup with all Blackwell fixes |
| `download_data.sh` | nuPlan v1.1 dataset parallel download + symlinks |
| `run_benchmark.sh` | Feature cache + 8-GPU DDP training |
| `launch_instance.sh` | EC2 launch with auto AMI detection |
| `patch_pluto_bf16.sh` | BF16 dtype patches for PLUTO model code |
| `profile_direct.py` | Per-phase profiling script (forward/backward/data) |
| `sweep_batch.sh` | Batch size sweep script |
| `RTX6000-BENCHMARK-REPORT.md` | This report |
