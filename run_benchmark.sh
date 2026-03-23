#!/bin/bash
set -ex

# Activate conda env
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate pluto

# === Blackwell / RTX PRO 6000 NCCL configuration ===
# CRITICAL: All 4 vars are required for DDP on Blackwell g7e instances.
#
# Disable P2P (broken on Blackwell: NVIDIA/nccl#1999, pytorch/pytorch#165727)
export NCCL_P2P_DISABLE=1
# No InfiniBand on g7e
export NCCL_IB_DISABLE=1
# Disable NVLink SHARP (can cause hangs on Blackwell)
export NCCL_NVLS_ENABLE=0
# Use correct network interface (g7e uses enp135s0, NOT eth0/ens5)
# "Bootstrap: no socket interface found" = this is wrong
export NCCL_SOCKET_IFNAME=enp135s0
# Debug level
export NCCL_DEBUG=WARN
# Improve DDP overlap efficiency
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Fix libstdc++ compatibility (Base DLAMI system lib is too old)
export LD_PRELOAD=/home/ubuntu/miniconda3/envs/pluto/lib/libstdc++.so.6

# Set paths - all on NVMe
export NUPLAN_DATA_ROOT=/nuplan/dataset
export NUPLAN_MAPS_ROOT=/nuplan/dataset/maps/nuplan-maps-v1.0
export NUPLAN_EXP_ROOT=/nuplan/exp

# Ensure pluto src is importable by Ray workers
export PYTHONPATH=/home/ubuntu/pluto:$PYTHONPATH

cd /home/ubuntu/pluto

echo "========================================="
echo "G7E Benchmark: 8x RTX PRO Server 6000"
echo "PyTorch $(python -c 'import torch; print(torch.__version__)') | CUDA $(python -c 'import torch; print(torch.version.cuda)') | NCCL $(python -c 'import torch; print(torch.cuda.nccl.version())')"
echo "========================================="

echo "========================================="
echo "Step 1: Feature Cache"
echo "Start: $(date)"
echo "========================================="

python run_training.py \
    py_func=cache +training=train_pluto \
    scenario_builder=nuplan \
    cache.cache_path=/nuplan/exp/cache_pluto \
    cache.cleanup_cache=true \
    scenario_filter=training_scenarios_1M \
    worker.threads_per_node=40 \
    2>&1 | tee /home/ubuntu/cache.log

echo "Cache done: $(date)"

# Verify cache has data
CACHE_COUNT=$(find /nuplan/exp/cache_pluto -name "*.gz" -o -name "*.pkl" 2>/dev/null | wc -l)
echo "Cache files: $CACHE_COUNT"
if [ "$CACHE_COUNT" -eq 0 ]; then
    echo "ERROR: Cache is empty! Aborting training."
    exit 1
fi

echo "========================================="
echo "Step 2: Training on 8x RTX PRO Server 6000"
echo "Precision: BF16-mixed (Blackwell Tensor Core 252 TFLOPS)"
echo "Batch size: 256 (32 per GPU, ~78% VRAM usage)"
echo "Start: $(date)"
echo "========================================="

# Monitor GPU utilization in background
nvidia-smi dmon -s umt -d 10 > /home/ubuntu/gpu_monitor.log 2>&1 &
GPU_MON_PID=$!

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_training.py \
    py_func=train +training=train_pluto \
    worker=single_machine_thread_pool worker.max_workers=32 \
    scenario_builder=nuplan \
    cache.cache_path=/nuplan/exp/cache_pluto \
    cache.use_cache_without_dataset=true \
    data_loader.params.batch_size=256 \
    data_loader.params.num_workers=20 \
    lr=1e-3 epochs=25 warmup_epochs=3 weight_decay=0.0001 \
    lightning.trainer.params.accelerator=gpu \
    lightning.trainer.params.devices=8 \
    lightning.trainer.params.strategy=ddp_find_unused_parameters_false \
    lightning.trainer.params.precision=bf16-mixed \
    wandb.mode=disabled \
    wandb.project=g7e-benchmark \
    wandb.name=pluto-8xRTXPRO6000 \
    2>&1 | tee /home/ubuntu/train.log

kill $GPU_MON_PID 2>/dev/null

echo "Training done: $(date)"
echo "========================================="
echo "=== GPU Monitor Summary ==="
tail -20 /home/ubuntu/gpu_monitor.log
echo ""
echo "=== Training Summary ==="
grep -E "Epoch|loss|time" /home/ubuntu/train.log | tail -30
