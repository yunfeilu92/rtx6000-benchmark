"""
Fix pin_memory: pin tensors AFTER DataLoader returns batch, BEFORE .cuda().
No need to change collate_fn.
"""
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


def recursive_pin_and_transfer(obj):
    """Pin memory then transfer to GPU in one pass."""
    if isinstance(obj, torch.Tensor):
        if not obj.is_pinned():
            return obj.pin_memory().cuda(non_blocking=True)
        return obj.cuda(non_blocking=True)
    elif isinstance(obj, dict):
        return {k: recursive_pin_and_transfer(v) for k, v in obj.items()}
    elif hasattr(obj, "data"):
        if isinstance(obj.data, dict):
            obj.data = recursive_pin_and_transfer(obj.data)
        elif isinstance(obj.data, torch.Tensor):
            if not obj.data.is_pinned():
                obj.data = obj.data.pin_memory().cuda(non_blocking=True)
            else:
                obj.data = obj.data.cuda(non_blocking=True)
        return obj
    return obj


def recursive_to_gpu(obj):
    """Current method: just .cuda(non_blocking=True) without pinning."""
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

# === A: Current (unpinned) ===
print("=== A: Current (.cuda non_blocking, data NOT pinned) ===")
times_a = []
for i in range(12):
    batch = next(it)
    f, t, _ = batch
    torch.cuda.synchronize()
    t0 = time.time()
    f = recursive_to_gpu(f)
    t = recursive_to_gpu(t)
    torch.cuda.synchronize()
    times_a.append(time.time() - t0)
    del f, t
    torch.cuda.empty_cache()
avg_a = sum(times_a[2:]) / len(times_a[2:])
print("  Avg: %.4fs" % avg_a)

# === B: Pin then transfer ===
print("")
print("=== B: Pin memory then .cuda(non_blocking) ===")
times_b = []
for i in range(12):
    batch = next(it)
    f, t, _ = batch
    torch.cuda.synchronize()
    t0 = time.time()
    f = recursive_pin_and_transfer(f)
    t = recursive_pin_and_transfer(t)
    torch.cuda.synchronize()
    times_b.append(time.time() - t0)
    del f, t
    torch.cuda.empty_cache()
avg_b = sum(times_b[2:]) / len(times_b[2:])
print("  Avg: %.4fs" % avg_b)

# === C: Pack all floats into 1 pinned buffer, single transfer ===
print("")
print("=== C: Pack + pin + single .cuda() ===")
times_c = []
for i in range(12):
    batch = next(it)
    f, t, _ = batch

    torch.cuda.synchronize()
    t0 = time.time()

    float_parts = []
    int_tensors = []
    def extract(obj):
        if isinstance(obj, torch.Tensor):
            if obj.is_floating_point():
                float_parts.append(obj.contiguous().view(-1))
            else:
                int_tensors.append(obj)
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
                int_tensors.append(obj.data)

    float_parts.clear()
    int_tensors.clear()
    extract(f)
    extract(t)

    packed = torch.cat(float_parts).pin_memory()
    packed_gpu = packed.cuda(non_blocking=True)
    for it_tensor in int_tensors:
        it_tensor.pin_memory().cuda(non_blocking=True)
    torch.cuda.synchronize()
    times_c.append(time.time() - t0)
    del packed, packed_gpu
    torch.cuda.empty_cache()

avg_c = sum(times_c[2:]) / len(times_c[2:])
print("  Avg: %.4fs" % avg_c)

# Summary
print("")
print("=" * 60)
print("RESULTS")
print("=" * 60)
print("  A. Unpinned .cuda():            %.4fs  (baseline)" % avg_a)
print("  B. Pin + .cuda():               %.4fs  (%.1fx)" % (avg_b, avg_a / avg_b))
print("  C. Pack + pin + single .cuda(): %.4fs  (%.1fx)" % (avg_c, avg_a / avg_c))
