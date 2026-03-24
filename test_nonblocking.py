"""Test: does non_blocking=True actually help vs blocking transfer?"""
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

def transfer_blocking(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cuda()  # blocking, no non_blocking
    elif isinstance(obj, dict):
        return {k: transfer_blocking(v) for k, v in obj.items()}
    elif hasattr(obj, 'data'):
        if isinstance(obj.data, dict):
            obj.data = transfer_blocking(obj.data)
        elif isinstance(obj.data, torch.Tensor):
            obj.data = obj.data.cuda()
        return obj
    return obj

def transfer_nonblocking(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cuda(non_blocking=True)
    elif isinstance(obj, dict):
        return {k: transfer_nonblocking(v) for k, v in obj.items()}
    elif hasattr(obj, 'data'):
        if isinstance(obj.data, dict):
            obj.data = transfer_nonblocking(obj.data)
        elif isinstance(obj.data, torch.Tensor):
            obj.data = obj.data.cuda(non_blocking=True)
        return obj
    return obj

it = iter(loader)

# Test 1: Blocking
print("=== TEST 1: .cuda() BLOCKING (no pin_memory in source) ===")
times = []
for i in range(10):
    batch = next(it)
    features, targets, _ = batch
    torch.cuda.synchronize()
    t0 = time.time()
    features = transfer_blocking(features)
    targets = transfer_blocking(targets)
    torch.cuda.synchronize()
    times.append(time.time() - t0)
avg_blocking = sum(times[2:]) / len(times[2:])
print("  Avg: %.4fs" % avg_blocking)

# Free GPU
del features, targets
torch.cuda.empty_cache()

# Test 2: Non-blocking (pin_memory already enabled in DataLoader)
print("")
print("=== TEST 2: .cuda(non_blocking=True) + pin_memory ===")
times = []
for i in range(10):
    batch = next(it)
    features, targets, _ = batch
    torch.cuda.synchronize()
    t0 = time.time()
    features = transfer_nonblocking(features)
    targets = transfer_nonblocking(targets)
    torch.cuda.synchronize()
    times.append(time.time() - t0)
avg_nonblocking = sum(times[2:]) / len(times[2:])
print("  Avg: %.4fs" % avg_nonblocking)

del features, targets
torch.cuda.empty_cache()

# Test 3: Non-blocking WITHOUT sync (measure just the queuing time)
print("")
print("=== TEST 3: .cuda(non_blocking=True) WITHOUT sync (queue only) ===")
times = []
for i in range(10):
    batch = next(it)
    features, targets, _ = batch
    t0 = time.time()
    features = transfer_nonblocking(features)
    targets = transfer_nonblocking(targets)
    # NO sync - just measure how fast we return to Python
    times.append(time.time() - t0)
avg_nosync = sum(times[2:]) / len(times[2:])
print("  Avg (queue only): %.4fs" % avg_nosync)
print("  This is the Python overhead of 38 .cuda() calls")

del features, targets
torch.cuda.empty_cache()

# Test 4: Non-blocking + overlap with dummy compute
print("")
print("=== TEST 4: Non-blocking + overlapped with GPU compute ===")
# Simulate: transfer batch N+1 while computing on batch N
dummy = torch.randn(1000, 1000, device='cuda')  # warmup

times_total = []
for i in range(10):
    batch = next(it)
    features, targets, _ = batch

    torch.cuda.synchronize()
    t0 = time.time()

    # Start async transfer
    features = transfer_nonblocking(features)
    targets = transfer_nonblocking(targets)

    # Simulate compute on GPU (matmul ~0.5s, similar to our forward+backward)
    for _ in range(500):
        dummy = torch.mm(dummy, dummy.t())

    torch.cuda.synchronize()
    total = time.time() - t0
    times_total.append(total)

avg_overlap = sum(times_total[2:]) / len(times_total[2:])

# Measure compute alone
times_compute = []
for i in range(5):
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(500):
        dummy = torch.mm(dummy, dummy.t())
    torch.cuda.synchronize()
    times_compute.append(time.time() - t0)
avg_compute = sum(times_compute[1:]) / len(times_compute[1:])

print("  Compute only: %.4fs" % avg_compute)
print("  Compute + transfer (overlapped): %.4fs" % avg_overlap)
print("  Transfer hidden: %.4fs (%.0f%% of transfer time hidden)" % (
    max(0, avg_nonblocking - (avg_overlap - avg_compute)),
    max(0, (1 - (avg_overlap - avg_compute) / avg_nonblocking) * 100)))

# Summary
print("")
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print("  Blocking .cuda():                    %.4fs" % avg_blocking)
print("  Non-blocking .cuda() + pin_memory:   %.4fs" % avg_nonblocking)
print("  Non-blocking queue time (no sync):   %.4fs" % avg_nosync)
print("  Speedup (blocking vs non-blocking):  %.2fx" % (avg_blocking / avg_nonblocking))
print("  Can overlap hide transfer?           %.0f%%" % (
    max(0, (1 - (avg_overlap - avg_compute) / avg_nonblocking) * 100)))
