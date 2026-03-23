#!/bin/bash
set -ex
# === Patch PLUTO for BF16-mixed precision on Blackwell ===
#
# Problem: Under torch.autocast(dtype=bfloat16), nn.Module outputs are BF16
# but torch.zeros() creates FP32 tensors. Assigning BF16 values into FP32
# tensors (or vice versa) via index operations causes:
#   RuntimeError: Index put requires the source and destination dtypes match
#
# Fix strategy:
# 1. torch.zeros() calls: inherit dtype from the autocast output that fills them
# 2. nn.Embedding.weight (always FP32): cast to tensor dtype when assigning
# 3. nn.Linear/Embedding outputs assigned into pre-allocated tensors: cast

PLUTO_DIR="${1:-/home/ubuntu/pluto}"
cd "$PLUTO_DIR"

# --- agent_encoder.py ---
# x_agent = torch.zeros(...) then x_agent[mask] = x_agent_tmp (BF16)
sed -i 's|x_agent = torch.zeros(bs \* A, self.dim, device=position.device)|x_agent = torch.zeros(bs * A, self.dim, device=position.device, dtype=x_agent_tmp.dtype)|' \
    src/models/pluto/modules/agent_encoder.py
# x_agent[:, 0] = x_ego where x_ego is from embedding (FP32) but x_agent is BF16
sed -i 's|x_agent\[:, 0\] = x_ego|x_agent[:, 0] = x_ego.to(x_agent.dtype)|' \
    src/models/pluto/modules/agent_encoder.py

# --- map_encoder.py ---
# x_speed_limit = torch.zeros(...) then filled with BF16 and FP32 values
sed -i 's|x_speed_limit = torch.zeros(bs, M, self.dim, device=x_polygon.device)|x_speed_limit = torch.zeros(bs, M, self.dim, device=x_polygon.device, dtype=x_polygon.dtype)|' \
    src/models/pluto/modules/map_encoder.py
# self.unknown_speed_emb.weight is always FP32, but x_speed_limit is now BF16
sed -i 's|x_speed_limit\[~polygon_has_speed_limit\] = self.unknown_speed_emb.weight|x_speed_limit[~polygon_has_speed_limit] = self.unknown_speed_emb.weight.to(x_speed_limit.dtype)|' \
    src/models/pluto/modules/map_encoder.py

# --- embedding.py ---
# x_features = torch.zeros(...) then x_features[mask] = x_valid (BF16)
sed -i 's|x_features = torch.zeros(bs, n, 256, device=device)|x_features = torch.zeros(bs, n, 256, device=device, dtype=x_valid.dtype)|' \
    src/models/pluto/layers/embedding.py
# res = torch.zeros(...) then res[mask] = x_features_valid (BF16)
sed -i 's|res = torch.zeros(bs, n, self.encoder_channel, device=device)|res = torch.zeros(bs, n, self.encoder_channel, device=device, dtype=x_features_valid.dtype)|' \
    src/models/pluto/layers/embedding.py

# Clear pycache
find src -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

echo "=== BF16 patches applied ==="
echo "Patched files:"
echo "  - src/models/pluto/modules/agent_encoder.py (zeros dtype + ego cast)"
echo "  - src/models/pluto/modules/map_encoder.py (zeros dtype + weight cast)"
echo "  - src/models/pluto/layers/embedding.py (2x zeros dtype)"
