"""Benchmark: individual vs packed CPU->GPU transfer."""
import sys, os, time, torch
sys.path.insert(0, "/home/ubuntu/pluto")
os.chdir("/home/ubuntu/pluto")
from hydra import compose, initialize_config_dir
initialize_config_dir(config_dir="/home/ubuntu/pluto/config", version_base="1.1")
cfg = compose(config_name="default_training", overrides=[
    "+training=train_pluto", "scenario_builder=nuplan",
    "cache.cache_path=/nuplan/exp/cache_pluto", "cache.use_cache_without_dataset=true",
    "data_loader.params.batch_size=256", "data_loader.params.num_workers=20",
])
from src.custom_training.custom_training_builder import build_training_engine
from nuplan.planning.script.builders.folder_builder import build_training_experiment_folder
from nuplan.planning.script.builders.worker_pool_builder import build_worker
build_training_experiment_folder(cfg)
worker = build_worker(cfg)
engine = build_training_engine(cfg, worker)
engine.datamodule.setup("fit")
loader = engine.datamodule.train_dataloader()

def recursive_to_gpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cuda(non_blocking=True)
    elif isinstance(obj, dict):
        return {k: recursive_to_gpu(v) for k, v in obj.items()}
    elif hasattr(obj, "data"):
        if isinstance(obj.data, dict):
            obj.data = recursive_to_gpu(obj.data)
        elif isinstance(obj.data, torch.Tensor):
            obj.data = obj.data.cuda(non_blocking=True)
        return obj
    return obj

it = iter(loader)

# Check pinned
batch = next(it)
features, targets, _ = batch
for v in features["feature"].data.values():
    if isinstance(v, torch.Tensor):
        print("is_pinned:", v.is_pinned(), "dtype:", v.dtype, "shape:", v.shape)
        break

# === A: Individual .cuda() ===
print("")
print("=== A: Individual .cuda() (38 calls) ===")
times = []
for i in range(12):
    batch = next(it)
    f, t, _ = batch
    torch.cuda.synchronize()
    t0 = time.time()
    f = recursive_to_gpu(f)
    t = recursive_to_gpu(t)
    torch.cuda.synchronize()
    times.append(time.time() - t0)
    del f, t
    torch.cuda.empty_cache()
avg_a = sum(times[2:]) / len(times[2:])
print("  Avg: %.4fs" % avg_a)

# === B: Packed + pin_memory ===
print("")
print("=== B: Packed + pin_memory (few calls) ===")
times_b = []
for i in range(12):
    batch = next(it)
    features, targets, _ = batch
    torch.cuda.synchronize()
    t0 = time.time()

    float_parts = []
    int_parts = []
    def extract(obj):
        if isinstance(obj, torch.Tensor):
            if obj.is_floating_point():
                float_parts.append(obj.contiguous().view(-1))
            else:
                int_parts.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                extract(v)
        elif hasattr(obj, "data") and isinstance(obj.data, dict):
            for v in obj.data.values():
                extract(v)
        elif hasattr(obj, "data") and isinstance(obj.data, torch.Tensor):
            if obj.data.is_floating_point():
                float_parts.append(obj.data.contiguous().view(-1))
            else:
                int_parts.append(obj.data)

    float_parts.clear()
    int_parts.clear()
    extract(features)
    extract(targets)

    packed = torch.cat(float_parts).pin_memory()
    packed_gpu = packed.cuda(non_blocking=True)
    for t in int_parts:
        t.cuda(non_blocking=True)
    torch.cuda.synchronize()
    times_b.append(time.time() - t0)
    del packed, packed_gpu
    torch.cuda.empty_cache()

avg_b = sum(times_b[2:]) / len(times_b[2:])
print("  Avg: %.4fs" % avg_b)

# === C: Just cost_maps (768MB) ===
print("")
print("=== C: cost_maps alone (768MB) ===")
times_c = []
for i in range(12):
    batch = next(it)
    f, _, _ = batch
    cm = f["feature"].data["cost_maps"]
    torch.cuda.synchronize()
    t0 = time.time()
    cm_gpu = cm.cuda(non_blocking=True)
    torch.cuda.synchronize()
    times_c.append(time.time() - t0)
    del cm_gpu
    torch.cuda.empty_cache()
avg_c = sum(times_c[2:]) / len(times_c[2:])
bw = 768e6 / avg_c / 1e9
print("  Avg: %.4fs (%.1f GB/s)" % (avg_c, bw))

# === D: All except cost_maps ===
print("")
print("=== D: Everything EXCEPT cost_maps (414MB, 37 tensors) ===")
times_d = []
for i in range(12):
    batch = next(it)
    features, targets, _ = batch
    torch.cuda.synchronize()
    t0 = time.time()
    # Transfer all except cost_maps individually
    for k, v in features["feature"].data.items():
        if k != "cost_maps" and isinstance(v, torch.Tensor):
            v.cuda(non_blocking=True)
        elif isinstance(v, dict):
            for vv in v.values():
                if isinstance(vv, torch.Tensor):
                    vv.cuda(non_blocking=True)
    for k, v in targets.items():
        if hasattr(v, "data") and isinstance(v.data, torch.Tensor):
            v.data.cuda(non_blocking=True)
    torch.cuda.synchronize()
    times_d.append(time.time() - t0)
    torch.cuda.empty_cache()
avg_d = sum(times_d[2:]) / len(times_d[2:])
print("  Avg: %.4fs" % avg_d)

# Summary
print("")
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print("  A. Individual (all 38):          %.4fs" % avg_a)
print("  B. Packed + pin_memory:          %.4fs (%.1fx)" % (avg_b, avg_a/avg_b))
print("  C. cost_maps only (768MB):       %.4fs (%.0f%% of A)" % (avg_c, avg_c/avg_a*100))
print("  D. All except cost_maps (37):    %.4fs (%.0f%% of A)" % (avg_d, avg_d/avg_a*100))
print("  C + D:                           %.4fs" % (avg_c + avg_d))
print("")
print("  cost_maps (1 tensor, 768MB) = %.0f%% of transfer time" % (avg_c/avg_a*100))
print("  other 37 tensors (414MB)    = %.0f%% of transfer time" % (avg_d/avg_a*100))
