#!/bin/bash
set -ex

echo "=== G7E RTX PRO 6000 Benchmark Setup ==="
echo "Instance: g7e.48xlarge (8x RTX PRO Server 6000, 96GB each)"
echo "Target: PyTorch 2.10.0 + CUDA 12.8 + Python 3.12 (Blackwell native)"
echo "Start time: $(date)"

# 1. System prep
sudo apt-get update -y && sudo apt-get install -y unzip wget git

# 2. Verify GPU & driver (need R570+ for Blackwell)
nvidia-smi
echo "CUDA compiler version:"
nvcc --version 2>/dev/null || echo "nvcc not in PATH (will use PyTorch bundled CUDA)"

# 3. Install Miniconda (Base DLAMI doesn't include conda)
if ! command -v conda &>/dev/null; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /home/ubuntu/miniconda3
    rm /tmp/miniconda.sh
fi
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh

# Accept conda ToS for default channels
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# 4. Create conda env with Python 3.12 for Blackwell compatibility
conda remove -n pluto --all -y 2>/dev/null || true
conda create -n pluto python=3.12 -y
conda activate pluto

# 5. Pin setuptools < 81 (pkg_resources required by pytorch-lightning 2.0.1)
pip install "setuptools>=70,<81"

# 6. Install PyTorch 2.10.0 with CUDA 12.8 (Blackwell sm_120 native, NCCL 2.27.5)
#    CRITICAL: PyTorch 2.7.0 ships NCCL 2.26.2 which hangs on Blackwell.
#    Must use >= 2.10.0 for NCCL >= 2.27.5.
pip install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 7. Upgrade cuBLAS (12.8.x has batched GEMM crash on Blackwell)
#    CRITICAL: Without this, torch.bmm() crashes with CUBLAS_STATUS_INVALID_VALUE
#    on ALL precisions (FP32/FP16/BF16).
pip install --force-reinstall --no-deps nvidia-cublas-cu12==12.9.1.4

# 8. Pin numpy < 2.0 (pytorch-lightning 2.0.1 uses np.Inf removed in numpy 2.0)
pip install "numpy<2.0"

# 9. Pin rich < 13 (PL 2.0.1 rich progress bar crashes with rich >= 13)
pip install "rich<13"

# 10. Install nuplan-devkit
cd /home/ubuntu
if [ ! -d nuplan-devkit ]; then
    git clone https://github.com/motional/nuplan-devkit.git
fi
cd nuplan-devkit
pip install -e .
# Install grpcio separately (old pinned version fails to build on Python 3.12)
pip install grpcio grpcio-tools
# Install remaining requirements (skip grpcio pin)
grep -v "grpcio" requirements.txt | pip install -r /dev/stdin || true
# Install missing deps not in requirements.txt
pip install aioboto3 aiofiles retry pyarrow geopandas pytest joblib numba \
    scikit-learn pyinstrument psutil bokeh docker pyquaternion scipy pandas \
    "protobuf>=4.25,<5" "ray[default]"
# Pin torchmetrics < 0.10 (compute_on_step removed in newer versions)
pip install "torchmetrics<0.10"
echo "=== NUPLAN_DEVKIT_DONE ==="

# 11. Install pluto
cd /home/ubuntu
if [ ! -d pluto ]; then
    git clone https://github.com/jchengai/pluto.git
fi
cd pluto

# Install pluto dependencies (skip setup_env.sh as it installs old PyTorch)
pip install pytorch-lightning==2.0.1 hydra-core==1.3.2 omegaconf
pip install natten wandb tensorboard timm

# === Pluto code patches for Python 3.12 + Blackwell ===

# Patch 1: Add missing __init__.py files (Python 3.12 needs explicit package init)
find src -type d -exec sh -c '[ ! -f "{}/__init__.py" ] && touch "{}/__init__.py"' \;

# Patch 2: Add exports to custom_training __init__.py
cat > src/custom_training/__init__.py << 'INITEOF'
from src.custom_training.custom_training_builder import (
    TrainingEngine,
    build_training_engine,
    update_config_for_training,
)
INITEOF

# Patch 3: NATTEN 0.21.x removed attn_drop param from NeighborhoodAttention1D
sed -i '/NeighborhoodAttention1D/,/)/{ /attn_drop=/d }' src/models/pluto/layers/embedding.py

# Patch 4: PyTorch 2.7+ removed verbose param from LRScheduler.__init__
sed -i 's/super(WarmupCosLR, self).__init__(optimizer, last_epoch, verbose)/super(WarmupCosLR, self).__init__(optimizer, last_epoch)/' src/optim/warmup_cos_lr.py

# Patch 5: BF16-mixed precision dtype fixes
# Under autocast, torch.zeros() creates FP32 but model outputs are BF16.
# Also nn.Embedding.weight stays FP32 and needs explicit casting.
# agent_encoder.py: zeros dtype + x_ego cast
sed -i 's|x_agent = torch.zeros(bs \* A, self.dim, device=position.device)|x_agent = torch.zeros(bs * A, self.dim, device=position.device, dtype=x_agent_tmp.dtype)|' src/models/pluto/modules/agent_encoder.py
sed -i 's|x_agent\[:, 0\] = x_ego|x_agent[:, 0] = x_ego.to(x_agent.dtype)|' src/models/pluto/modules/agent_encoder.py
# map_encoder.py: zeros dtype + embedding weight cast
sed -i 's|x_speed_limit = torch.zeros(bs, M, self.dim, device=x_polygon.device)|x_speed_limit = torch.zeros(bs, M, self.dim, device=x_polygon.device, dtype=x_polygon.dtype)|' src/models/pluto/modules/map_encoder.py
sed -i 's|x_speed_limit\[~polygon_has_speed_limit\] = self.unknown_speed_emb.weight|x_speed_limit[~polygon_has_speed_limit] = self.unknown_speed_emb.weight.to(x_speed_limit.dtype)|' src/models/pluto/modules/map_encoder.py
# embedding.py: 2x zeros dtype
sed -i 's|x_features = torch.zeros(bs, n, 256, device=device)|x_features = torch.zeros(bs, n, 256, device=device, dtype=x_valid.dtype)|' src/models/pluto/layers/embedding.py
sed -i 's|res = torch.zeros(bs, n, self.encoder_channel, device=device)|res = torch.zeros(bs, n, self.encoder_channel, device=device, dtype=x_features_valid.dtype)|' src/models/pluto/layers/embedding.py

echo "=== PLUTO_DONE ==="

# 12. Create data directories on NVMe (14TB available on g7e.48xlarge)
sudo mkdir -p /opt/dlami/nvme/nuplan/dataset
sudo mkdir -p /opt/dlami/nvme/nuplan/exp
sudo ln -sf /opt/dlami/nvme/nuplan /nuplan
sudo chown -R ubuntu:ubuntu /opt/dlami/nvme/nuplan

# 13. Verify installation
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'cuDNN version: {torch.backends.cudnn.version()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name}, {props.total_memory / 1024**3:.1f} GB, SM {props.major}.{props.minor}')
print(f'NCCL version: {torch.cuda.nccl.version()}')
print(f'BF16 supported: {torch.cuda.is_bf16_supported()}')

# Verify cuBLAS batched GEMM works (critical Blackwell fix)
a = torch.randn(8, 32, 128, device='cuda:0')
b = torch.randn(8, 128, 64, device='cuda:0')
c = torch.bmm(a, b)
print(f'cuBLAS batched GEMM OK: {c.shape}')
"

# Verify pluto imports
cd /home/ubuntu/pluto
PYTHONPATH=/home/ubuntu/pluto:\$PYTHONPATH python -c "
from src.custom_training import TrainingEngine
from src.models.pluto.pluto_model import PlanningModel
print('Pluto imports OK')
"

echo "=== ALL_DONE ==="
echo "End time: $(date)"
