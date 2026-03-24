#!/bin/bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate pluto

export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_SOCKET_IFNAME=enp135s0 NCCL_DEBUG=WARN
export CUDA_DEVICE_MAX_CONNECTIONS=1
export LD_PRELOAD=/home/ubuntu/miniconda3/envs/pluto/lib/libstdc++.so.6
export PYTHONPATH=/home/ubuntu/pluto:$PYTHONPATH
export NUPLAN_DATA_ROOT=/nuplan/dataset
export NUPLAN_MAPS_ROOT=/nuplan/dataset/maps/nuplan-maps-v1.0
export NUPLAN_EXP_ROOT=/nuplan/exp

cd /home/ubuntu/pluto

for BS in 128 256 512; do
    echo ""
    echo "################################################"
    echo "## 8-GPU BF16 BATCH=$BS - 3 EPOCHS"
    echo "################################################"

    START_TS=$(date +%s)

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 1800 python run_training.py \
        py_func=train +training=train_pluto \
        worker=single_machine_thread_pool worker.max_workers=32 \
        scenario_builder=nuplan \
        cache.cache_path=/nuplan/exp/cache_pluto \
        cache.use_cache_without_dataset=true \
        data_loader.params.batch_size=$BS \
        data_loader.params.num_workers=20 \
        data_loader.params.pin_memory=true \
        lr=1e-3 epochs=3 warmup_epochs=0 weight_decay=0.0001 \
        lightning.trainer.params.accelerator=gpu \
        lightning.trainer.params.devices=8 \
        lightning.trainer.params.strategy=ddp_find_unused_parameters_false \
        lightning.trainer.params.precision=bf16-mixed \
        wandb.mode=disabled \
        "++lightning.trainer.params.num_sanity_val_steps=0" \
        2>/dev/null

    END_TS=$(date +%s)
    ELAPSED=$((END_TS - START_TS))

    LATEST=$(ls -td /nuplan/exp/exp/training/pluto/2026.03.2* 2>/dev/null | head -1)
    EPOCHS_DONE=0
    if [ -n "$LATEST" ]; then
        EPOCHS_DONE=$(ls "$LATEST"/checkpoints/epoch*.ckpt 2>/dev/null | wc -l)
    fi

    echo ""
    echo "=== RESULT: BS=$BS | ${ELAPSED}s total | $EPOCHS_DONE epochs ==="
    if [ -n "$LATEST" ]; then
        ls -lt --full-time "$LATEST"/checkpoints/epoch*.ckpt 2>/dev/null | awk '{print $6, $7, $9}'
    fi

    # Record GPU memory from last step
    nvidia-smi --query-gpu=index,memory.used,power.draw --format=csv,noheader

    # Kill everything for next test
    kill -9 $(fuser /dev/nvidia0 /dev/nvidia1 /dev/nvidia2 /dev/nvidia3 /dev/nvidia4 /dev/nvidia5 /dev/nvidia6 /dev/nvidia7 2>/dev/null) 2>/dev/null
    sleep 5
done

echo ""
echo "========== SWEEP DONE =========="
