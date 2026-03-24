"""
Fix: pin_memory doesn't work with nuPlan's custom Feature objects.
Benchmark before/after to measure real impact.
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


def recursive_pin(obj):
    """Pin memory recursively, handling custom Feature objects."""
    if isinstance(obj, torch.Tensor):
        return obj.pin_memory()
    elif isinstance(obj, dict):
        return {k: recursive_pin(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(recursive_pin(v) for v in obj)
    elif hasattr(obj, "data"):
        if isinstance(obj.data, dict):
            obj.data = {k: recursive_pin(v) for k, v in obj.data.items()}
        elif isinstance(obj.data, torch.Tensor):
            obj.data = obj.data.pin_memory()
        return obj
    return obj


def custom_collate_with_pin(batch_list):
    """Use default collate, then pin all tensors including those inside Feature objects."""
    from torch.utils.data._utils.collate import default_collate
    batch = default_collate(batch_list)
    return recursive_pin(batch)


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


# === Test A: Current (pin_memory=True but not working) ===
print("=== A: Current DataLoader (pin_memory=True but broken) ===")
loader_broken = engine.datamodule.train_dataloader()
it = iter(loader_broken)

# Verify: not pinned
batch = next(it)
for v in batch[0]["feature"].data.values():
    if isinstance(v, torch.Tensor):
        print("  is_pinned:", v.is_pinned())
        break

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
print("  Transfer: %.4fs" % avg_a)
del loader_broken

# === Test B: Fixed pin_memory via custom collate ===
print("")
print("=== B: Fixed DataLoader (custom collate with recursive pin) ===")
from torch.utils.data import DataLoader
orig_loader = engine.datamodule.train_dataloader()
loader_fixed = DataLoader(
    orig_loader.dataset,
    batch_size=orig_loader.batch_size,
    num_workers=20,
    pin_memory=False,  # We pin manually in collate
    shuffle=False,
    collate_fn=custom_collate_with_pin,
)
it = iter(loader_fixed)

# Verify: pinned!
batch = next(it)
for v in batch[0]["feature"].data.values():
    if isinstance(v, torch.Tensor):
        print("  is_pinned:", v.is_pinned())
        break

times_b = []
for i in range(12):
    batch = next(it)
    f, t, _ = batch
    torch.cuda.synchronize()
    t0 = time.time()
    f = recursive_to_gpu(f)
    t = recursive_to_gpu(t)
    torch.cuda.synchronize()
    times_b.append(time.time() - t0)
    del f, t
    torch.cuda.empty_cache()
avg_b = sum(times_b[2:]) / len(times_b[2:])
print("  Transfer: %.4fs" % avg_b)
del loader_fixed

# === Test C: Fixed pin + packed (best possible) ===
print("")
print("=== C: Fixed pin + packed (1 large transfer) ===")
orig_loader = engine.datamodule.train_dataloader()
loader_fixed2 = DataLoader(
    orig_loader.dataset,
    batch_size=orig_loader.batch_size,
    num_workers=20,
    pin_memory=False,
    shuffle=False,
    collate_fn=custom_collate_with_pin,
)
it = iter(loader_fixed2)

times_c = []
for i in range(12):
    batch = next(it)
    f, t, _ = batch

    # Pack all float tensors
    float_parts = []
    def extract_floats(obj):
        if isinstance(obj, torch.Tensor) and obj.is_floating_point():
            float_parts.append(obj.contiguous().view(-1))
        elif isinstance(obj, dict):
            for v in obj.values():
                extract_floats(v)
        elif hasattr(obj, "data") and isinstance(obj.data, dict):
            for v in obj.data.values():
                extract_floats(v)

    torch.cuda.synchronize()
    t0 = time.time()
    float_parts.clear()
    extract_floats(f)
    extract_floats(t)
    # Already pinned from collate, no need to re-pin
    packed = torch.cat(float_parts)
    packed_gpu = packed.cuda(non_blocking=True)
    torch.cuda.synchronize()
    times_c.append(time.time() - t0)
    del packed, packed_gpu
    torch.cuda.empty_cache()

avg_c = sum(times_c[2:]) / len(times_c[2:])
print("  Transfer: %.4fs" % avg_c)

# Summary
print("")
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print("  A. Broken pin_memory (current):  %.4fs" % avg_a)
print("  B. Fixed pin_memory:             %.4fs (%.1fx speedup)" % (avg_b, avg_a/avg_b))
print("  C. Fixed pin + packed:           %.4fs (%.1fx speedup)" % (avg_c, avg_a/avg_c))
print("")
print("  Pin_memory fix alone saves:      %.4fs (%.0f%%)" % (avg_a - avg_b, (1 - avg_b/avg_a)*100))
print("  Pin + pack saves:                %.4fs (%.0f%%)" % (avg_a - avg_c, (1 - avg_c/avg_a)*100))
