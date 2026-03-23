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

### Training Config
| Parameter | Value |
|---|---|
| Batch Size | 256 (32 per GPU) |
| GPUs | 8 (DDP) |
| Epochs | 25 |
| Learning Rate | 1e-3 (Cosine + 3 epoch warmup) |
| Precision | FP32 |
| Strategy | DDP (find_unused_parameters=false) |
| DataLoader Workers | 20 |

## Results

### Feature Cache (Step 1)
| Metric | Value |
|---|---|
| Duration | **2h 33m** |
| Scenarios | 25,000 |
| Ray Workers | 40 |
| Cache Size | ~460 GB |

### Training (Step 2)

#### Batch Size = 64 (8 per GPU, FP32)
| Metric | Value |
|---|---|
| Epoch Time | **~22 min** |
| Steps per Epoch | 1,515 |
| Throughput | ~4,500 samples/sec |
| GPU Utilization | 96-100% |
| VRAM Usage | ~25 GB / 95 GB (26%) |
| Power Draw | ~135W / 600W (22%) |
| Temperature | 30-32°C |

#### Batch Size = 256 (32 per GPU, FP32)
| Metric | Value |
|---|---|
| Epoch Time | **~28 min** |
| Steps per Epoch | 379 |
| Throughput | ~4,600 samples/sec |
| GPU Utilization | 100% |
| VRAM Usage | ~65-91 GB / 95 GB (68-95%) |
| Power Draw | ~100W / 600W (17%) |
| Temperature | 30-32°C |

## Profiling Analysis

Detailed profiling on single GPU, batch=256, FP32, 20 steps:

| Phase | Time per Step | Percentage |
|---|---|---|
| Data transfer (CPU->GPU) | 0.316s | 8.4% |
| **Forward pass** | **1.385s** | **36.9%** |
| Loss compute | 0.003s | 0.1% |
| **Backward pass** | **2.037s** | **54.3%** |
| Optimizer step | 0.008s | 0.2% |
| **Total** | **3.749s** | 100% |
| **Throughput** | **68 samples/sec** | (single GPU) |

### Key Findings

1. **Backward dominates (54%)** — PLUTO uses masked indexing, `grid_sample`, Neighborhood Attention, which have expensive backward ops
2. **Forward is 37%** — as expected for a small model
3. **Data transfer is 8.4%** — 0.32s to move 256 samples from CPU to GPU
4. **Loss and optimizer are negligible** — model is tiny (4.1M params)

### Why BF16 Doesn't Help This Model

We tested BF16-mixed precision and found **zero speedup** over FP32:
- PLUTO's 4.1M parameters create small matrix operations that can't saturate Tensor Cores
- Tensor Cores need large matmuls (dimensions >> 256) to achieve speedup
- GPU is at 17% TDP in FP32 — it's not compute-bound
- BF16 autocast overhead (dtype casting, GradScaler) cancels out any theoretical gain

### Throughput Is Constant Across Batch Sizes

| Batch Size | Epoch Time | Throughput |
|---|---|---|
| 64 | 22 min | ~4,500 samples/sec |
| 256 | 28 min | ~4,600 samples/sec |

Throughput barely changes because the bottleneck is DDP communication over PCIe (no NVLink, P2P disabled) and CPU-GPU data transfer, not GPU compute.

### Optimization Assessment

| Optimization | Applicable? | Expected Impact |
|---|---|---|
| torch.compile | Blocked — Triton segfaults on SM 12.0 | N/A |
| pin_memory=True | Yes | 2-4% (reduce data transfer) |
| BF16-mixed | Works but no speedup | 0% for small models |
| Larger batch (512) | Works but throughput constant | ~0% |
| Gradient accumulation | Adds overhead | Negative |
| More workers (32) | Loading not bottleneck | ~0% |
| CUDA Graphs | Blocked — dynamic shapes + SM 12.0 | N/A |

## Blackwell Compatibility Issues (Critical)

### 1. NCCL 2.26.2 Completely Broken
PyTorch 2.7.0 ships NCCL 2.26.2 which **hangs indefinitely** on Blackwell during any collective operation (AllReduce, AllGather). Even single-GPU `dist.init_process_group()` hangs.

**Fix**: Use PyTorch >= 2.10.0 (ships NCCL 2.27.5).

### 2. cuBLAS Batched GEMM Crash
cuBLAS 12.8.x crashes with `CUBLAS_STATUS_INVALID_VALUE` on `torch.bmm()`, `cublasSgemmStridedBatched`, and all batched GEMM ops. Affects **all precisions** (FP32/FP16/BF16).

**Fix**: `pip install --force-reinstall --no-deps nvidia-cublas-cu12==12.9.1.4`

### 3. NCCL Socket Interface
g7e instances use network interface `enp135s0`, not `eth0` or `ens5`. Setting `NCCL_SOCKET_IFNAME=eth0` causes "Bootstrap: no socket interface found".

**Fix**: `export NCCL_SOCKET_IFNAME=enp135s0`

### 4. Required NCCL Environment Variables
```bash
export NCCL_P2P_DISABLE=1     # P2P broken on Blackwell (NVIDIA/nccl#1999)
export NCCL_IB_DISABLE=1      # No InfiniBand on g7e
export NCCL_NVLS_ENABLE=0     # NVLink SHARP causes hangs
export NCCL_SOCKET_IFNAME=enp135s0
```

### 5. BF16/FP16 Mixed Precision Dtype Mismatches
PLUTO code has two types of dtype issues under autocast:
1. `torch.zeros()` without dtype creates FP32 tensors that receive BF16 autocast outputs
2. `nn.Embedding.weight` stays FP32 but is assigned into BF16 tensors

**Fix**: `patch_pluto_bf16.sh` patches 3 files (6 locations):
- `agent_encoder.py`: `torch.zeros()` dtype + `x_ego` cast
- `map_encoder.py`: `torch.zeros()` dtype + `unknown_speed_emb.weight` cast
- `embedding.py`: 2x `torch.zeros()` dtype

### 6. PLUTO Code Patches Required
- Missing `__init__.py` in all `src/` subdirectories (Python 3.12 compat)
- NATTEN 0.21.x removed `attn_drop` parameter
- PyTorch 2.7+ removed `verbose` from `LRScheduler.__init__`
- nuplan data directory structure needs symlinks

### 7. Triton / torch.compile Not Ready for SM 12.0
[PyTorch issue #176426](https://github.com/pytorch/pytorch/issues/176426): Triton kernels with 2+ `tl.load()` calls segfault at runtime on SM 12.0. This blocks `torch.compile` and CUDA Graphs.

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
| FP8 Tensor | 733 TFLOPS | 504 TFLOPS (L40S wins) |
| FP4 Tensor | N/A | 2,015 TFLOPS (exclusive) |
| VRAM | 48 GB GDDR6 | 96 GB GDDR7 (2x) |
| Memory BW | 864 GB/s | 1,600 GB/s (1.85x) |
| MIG | No | Yes (4x 24GB) |
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
| `RTX6000-BENCHMARK-REPORT.md` | This report |
