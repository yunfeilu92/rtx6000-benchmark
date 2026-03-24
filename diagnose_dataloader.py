"""Diagnose DataLoader bottleneck: tensor count, sizes, transfer overhead."""
import sys, os, time, torch
sys.path.insert(0, '/home/ubuntu/pluto')
os.chdir('/home/ubuntu/pluto')
from hydra import compose, initialize_config_dir
initialize_config_dir(config_dir='/home/ubuntu/pluto/config', version_base='1.1')
cfg = compose(config_name='default_training', overrides=[
    '+training=train_pluto', 'scenario_builder=nuplan',
    'cache.cache_path=/nuplan/exp/cache_pluto', 'cache.use_cache_without_dataset=true',
    'data_loader.params.batch_size=256', 'data_loader.params.num_workers=20',
])
from src.custom_training.custom_training_builder import build_training_engine
from nuplan.planning.script.builders.folder_builder import build_training_experiment_folder
from nuplan.planning.script.builders.worker_pool_builder import build_worker
build_training_experiment_folder(cfg)
worker = build_worker(cfg)
engine = build_training_engine(cfg, worker)
engine.datamodule.setup('fit')
loader = engine.datamodule.train_dataloader()

# =====================================================
# TEST 1: What's inside a batch?
# =====================================================
print("=" * 60)
print("TEST 1: Batch structure analysis")
print("=" * 60)

batch = next(iter(loader))
features, targets, scenarios = batch

tensor_info = []
def scan_tensors(obj, prefix=''):
    if isinstance(obj, torch.Tensor):
        tensor_info.append((prefix, obj.shape, obj.dtype, obj.nelement() * obj.element_size()))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            scan_tensors(v, prefix + '/' + k)
    elif hasattr(obj, 'data') and isinstance(obj.data, dict):
        for k, v in obj.data.items():
            scan_tensors(v, prefix + '/' + k)
    elif hasattr(obj, 'data') and isinstance(obj.data, torch.Tensor):
        scan_tensors(obj.data, prefix + '.data')

scan_tensors(features, 'features')
scan_tensors(targets, 'targets')

total_bytes = sum(t[3] for t in tensor_info)
float_tensors = [t for t in tensor_info if 'float' in str(t[2])]
int_tensors = [t for t in tensor_info if 'int' in str(t[2]) or 'bool' in str(t[2])]

print("Total tensors: %d" % len(tensor_info))
print("  Float tensors: %d (%.1f MB)" % (len(float_tensors), sum(t[3] for t in float_tensors) / 1e6))
print("  Int/Bool tensors: %d (%.1f MB)" % (len(int_tensors), sum(t[3] for t in int_tensors) / 1e6))
print("  Total size: %.1f MB" % (total_bytes / 1e6))
print("")
print("Per-tensor breakdown:")
for name, shape, dtype, nbytes in sorted(tensor_info, key=lambda x: -x[3]):
    print("  %6.2f MB  %-8s  %-30s  %s" % (nbytes/1e6, dtype, str(list(shape)), name))

# =====================================================
# TEST 2: DataLoader __next__() speed (CPU only)
# =====================================================
print("")
print("=" * 60)
print("TEST 2: DataLoader __next__() speed (CPU, no GPU)")
print("=" * 60)

it = iter(loader)
times = []
for i in range(20):
    t0 = time.time()
    batch = next(it)
    times.append(time.time() - t0)

print("  First batch: %.4fs" % times[0])
print("  Avg (skip 3): %.4fs (%.0f batches/sec)" % (
    sum(times[3:]) / len(times[3:]),
    len(times[3:]) / sum(times[3:])))

# =====================================================
# TEST 3: Individual .cuda() transfer (current method)
# =====================================================
print("")
print("=" * 60)
print("TEST 3: Current transfer method (individual .cuda() calls)")
print("=" * 60)

def recursive_to_gpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cuda(non_blocking=True)
    elif isinstance(obj, dict):
        return {k: recursive_to_gpu(v) for k, v in obj.items()}
    elif hasattr(obj, 'data'):
        if isinstance(obj.data, dict):
            obj.data = recursive_to_gpu(obj.data)
        elif isinstance(obj.data, torch.Tensor):
            obj.data = obj.data.cuda(non_blocking=True)
        return obj
    return obj

it = iter(loader)
times = []
for i in range(10):
    batch = next(it)
    features, targets, scenarios = batch
    torch.cuda.synchronize()
    t0 = time.time()
    features = recursive_to_gpu(features)
    targets = recursive_to_gpu(targets)
    torch.cuda.synchronize()
    times.append(time.time() - t0)

avg_individual = sum(times[2:]) / len(times[2:])
bw = total_bytes / avg_individual / 1e9
print("  Avg transfer: %.4fs" % avg_individual)
print("  Effective BW: %.1f GB/s (PCIe 5.0 max: 64 GB/s, %.1f%% util)" % (bw, bw/64*100))

# =====================================================
# TEST 4: Packed transfer (single .cuda() call)
# =====================================================
print("")
print("=" * 60)
print("TEST 4: Packed transfer (1 .cuda() call)")
print("=" * 60)

it = iter(loader)
times_pack = []
times_xfer = []
for i in range(10):
    batch = next(it)
    features, targets, scenarios = batch

    # Pack
    float_list = []
    def extract_floats(obj):
        if isinstance(obj, torch.Tensor) and obj.is_floating_point():
            float_list.append(obj.contiguous().view(-1))
        elif isinstance(obj, dict):
            for v in obj.values(): extract_floats(v)
        elif hasattr(obj, 'data') and isinstance(obj.data, dict):
            for v in obj.data.values(): extract_floats(v)
        elif hasattr(obj, 'data') and isinstance(obj.data, torch.Tensor) and obj.data.is_floating_point():
            float_list.append(obj.data.contiguous().view(-1))

    t0 = time.time()
    float_list.clear()
    extract_floats(features)
    extract_floats(targets)
    packed = torch.cat(float_list).pin_memory()
    t_pack = time.time() - t0

    torch.cuda.synchronize()
    t1 = time.time()
    packed_gpu = packed.cuda(non_blocking=True)
    torch.cuda.synchronize()
    t_xfer = time.time() - t1

    times_pack.append(t_pack)
    times_xfer.append(t_xfer)

avg_pack = sum(times_pack[2:]) / len(times_pack[2:])
avg_xfer = sum(times_xfer[2:]) / len(times_xfer[2:])
total_packed = avg_pack + avg_xfer
bw_packed = total_bytes / avg_xfer / 1e9

print("  CPU pack time: %.4fs" % avg_pack)
print("  GPU transfer:  %.4fs" % avg_xfer)
print("  Total:         %.4fs" % total_packed)
print("  Effective BW:  %.1f GB/s (%.1f%% PCIe util)" % (bw_packed, bw_packed/64*100))

# =====================================================
# SUMMARY
# =====================================================
print("")
print("=" * 60)
print("SUMMARY")
print("=" * 60)
speedup = avg_individual / total_packed
print("  Current (individual .cuda()):  %.4fs" % avg_individual)
print("  Packed (single .cuda()):       %.4fs" % total_packed)
print("  Speedup:                       %.1fx" % speedup)
print("  Time saved per step:           %.4fs (%.0f%%)" % (
    avg_individual - total_packed,
    (1 - total_packed / avg_individual) * 100))
