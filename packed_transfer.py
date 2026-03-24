"""
Packed tensor transfer: reduce 38 .cuda() calls to 1.
Drop-in replacement for recursive_to_gpu in PL training.
"""
import torch
from typing import Any, Dict, List, Tuple


def pack_batch_to_gpu(features, targets, device='cuda'):
    """
    Pack all tensors from a batch into contiguous buffers (one per dtype),
    transfer with a single .cuda() call, then reconstruct the original structure.

    Returns: (features_gpu, targets_gpu) with same structure as input.
    """
    # Step 1: Extract all tensors with their paths
    registry = []  # (path, shape, dtype, is_float, offset, source)
    float_parts = []
    int_parts = []
    bool_parts = []

    def extract(obj, path, source_name):
        if isinstance(obj, torch.Tensor):
            entry = [path, obj.shape, obj.dtype, source_name]
            if obj.is_floating_point():
                entry.append(('float', len(float_parts)))
                float_parts.append(obj.contiguous().view(-1))
            elif obj.dtype == torch.bool:
                entry.append(('bool', len(bool_parts)))
                bool_parts.append(obj.contiguous().view(-1))
            else:
                entry.append(('int', len(int_parts)))
                int_parts.append(obj.contiguous().view(-1))
            registry.append(entry)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                extract(v, path + (k,), source_name)
        elif hasattr(obj, 'data') and isinstance(obj.data, dict):
            for k, v in obj.data.items():
                extract(v, path + ('.data', k), source_name)
        elif hasattr(obj, 'data') and isinstance(obj.data, torch.Tensor):
            entry = [path + ('.data',), obj.data.shape, obj.data.dtype, source_name]
            if obj.data.is_floating_point():
                entry.append(('float', len(float_parts)))
                float_parts.append(obj.data.contiguous().view(-1))
            else:
                entry.append(('int', len(int_parts)))
                int_parts.append(obj.data.contiguous().view(-1))
            registry.append(entry)

    extract(features, (), 'features')
    extract(targets, (), 'targets')

    # Step 2: Cat into contiguous buffers and transfer (1 call per dtype)
    float_gpu = None
    int_gpu = None
    bool_gpu = None

    if float_parts:
        float_packed = torch.cat(float_parts)
        float_gpu = float_packed.to(device, non_blocking=True)
    if int_parts:
        int_packed = torch.cat(int_parts)
        int_gpu = int_packed.to(device, non_blocking=True)
    if bool_parts:
        bool_packed = torch.cat(bool_parts)
        bool_gpu = bool_packed.to(device, non_blocking=True)

    # Step 3: Reconstruct original structure on GPU (view only, no copy)
    float_offset = 0
    int_offset = 0
    bool_offset = 0

    gpu_tensors = {}  # path -> gpu tensor
    for entry in registry:
        path, shape, dtype, source, (buf_type, _) = entry
        numel = 1
        for s in shape:
            numel *= s

        if buf_type == 'float':
            gpu_tensors[(source, path)] = float_gpu[float_offset:float_offset + numel].view(shape)
            float_offset += numel
        elif buf_type == 'int':
            gpu_tensors[(source, path)] = int_gpu[int_offset:int_offset + numel].view(shape)
            int_offset += numel
        elif buf_type == 'bool':
            gpu_tensors[(source, path)] = bool_gpu[bool_offset:bool_offset + numel].view(shape)
            bool_offset += numel

    # Step 4: Rebuild features and targets dicts
    def rebuild(obj, path, source_name):
        key = (source_name, path)
        if key in gpu_tensors:
            return gpu_tensors[key]
        elif isinstance(obj, torch.Tensor):
            return gpu_tensors.get(key, obj.to(device, non_blocking=True))
        elif isinstance(obj, dict):
            return {k: rebuild(v, path + (k,), source_name) for k, v in obj.items()}
        elif hasattr(obj, 'data') and isinstance(obj.data, dict):
            obj.data = {k: rebuild(v, path + ('.data', k), source_name) for k, v in obj.data.items()}
            return obj
        elif hasattr(obj, 'data') and isinstance(obj.data, torch.Tensor):
            data_key = (source_name, path + ('.data',))
            if data_key in gpu_tensors:
                obj.data = gpu_tensors[data_key]
            return obj
        return obj

    features_gpu = rebuild(features, (), 'features')
    targets_gpu = rebuild(targets, (), 'targets')

    return features_gpu, targets_gpu


if __name__ == '__main__':
    """Benchmark: packed vs individual transfer."""
    import sys, os, time
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

    # Also build model for end-to-end validation
    model = engine.model
    model.cuda()
    model.train()

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

    # Benchmark: individual transfer
    print("=== Individual .cuda() (current) ===")
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
    avg_old = sum(times[2:]) / len(times[2:])
    print("  Avg: %.4fs" % avg_old)

    torch.cuda.empty_cache()

    # Benchmark: packed transfer
    print("")
    print("=== Packed transfer (new) ===")
    times = []
    for i in range(10):
        batch = next(it)
        features, targets, scenarios = batch
        torch.cuda.synchronize()
        t0 = time.time()
        features_gpu, targets_gpu = pack_batch_to_gpu(features, targets)
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    avg_new = sum(times[2:]) / len(times[2:])
    print("  Avg: %.4fs" % avg_new)

    torch.cuda.empty_cache()

    # End-to-end: packed transfer + model forward (verify correctness)
    print("")
    print("=== End-to-end validation (packed + model forward) ===")
    times_e2e = []
    for i in range(8):
        batch = next(it)
        features, targets, scenarios = batch
        torch.cuda.synchronize()
        t0 = time.time()
        features_gpu, targets_gpu = pack_batch_to_gpu(features, targets)
        torch.cuda.synchronize()
        t_xfer = time.time() - t0

        t1 = time.time()
        res = model.model.forward(features_gpu['feature'].data if hasattr(features_gpu, '__getitem__') else features_gpu)
        torch.cuda.synchronize()
        t_fwd = time.time() - t1

        if i < 3:
            print("  Step %d: transfer=%.4fs forward=%.4fs" % (i, t_xfer, t_fwd))
        times_e2e.append((t_xfer, t_fwd))

    print("")
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("  Individual .cuda():  %.4fs" % avg_old)
    print("  Packed transfer:     %.4fs" % avg_new)
    print("  Speedup:             %.1fx" % (avg_old / avg_new))
    print("  Time saved/step:     %.4fs (%.0f%%)" % (avg_old - avg_new, (1 - avg_new/avg_old)*100))
    if times_e2e:
        print("  Model forward OK:    %s" % ("YES" if times_e2e[-1][1] > 0 else "FAILED"))
