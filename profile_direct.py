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

lt = engine.model
lt.cuda()
lt.train()
datamodule = engine.datamodule
datamodule.setup("fit")
loader = datamodule.train_dataloader()
optimizer = torch.optim.AdamW(lt.parameters(), lr=1e-3)

def recursive_to_gpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cuda(non_blocking=True)
    elif isinstance(obj, dict):
        return {k: recursive_to_gpu(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(recursive_to_gpu(v) for v in obj)
    elif hasattr(obj, "data"):
        if isinstance(obj.data, dict):
            obj.data = recursive_to_gpu(obj.data)
        elif isinstance(obj.data, torch.Tensor):
            obj.data = obj.data.cuda(non_blocking=True)
        return obj
    return obj

def move_batch_to_gpu(batch):
    features, targets, scenarios = batch
    features = recursive_to_gpu(features)
    targets = recursive_to_gpu(targets)
    return features, targets, scenarios

print("Profiling: 20 steps, batch=256, FP32, single GPU")
records = []
for step, batch in enumerate(loader):
    if step >= 20:
        break

    torch.cuda.synchronize()
    t0 = time.time()
    features, targets, scenarios = move_batch_to_gpu(batch)
    torch.cuda.synchronize()
    t_data = time.time() - t0

    torch.cuda.synchronize()
    t1 = time.time()
    res = lt.forward(features["feature"].data)
    torch.cuda.synchronize()
    t_fwd = time.time() - t1

    torch.cuda.synchronize()
    t2 = time.time()
    # _compute_objectives expects features data (has 'agent', 'map', etc), not targets
    feat_data = features["feature"].data
    objectives = lt._compute_objectives(res, feat_data)
    loss = objectives["loss"]
    torch.cuda.synchronize()
    t_loss = time.time() - t2

    torch.cuda.synchronize()
    t3 = time.time()
    loss.backward()
    torch.cuda.synchronize()
    t_bwd = time.time() - t3

    torch.cuda.synchronize()
    t4 = time.time()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    t_opt = time.time() - t4

    total = t_data + t_fwd + t_loss + t_bwd + t_opt
    records.append({"data": t_data, "fwd": t_fwd, "loss": t_loss, "bwd": t_bwd, "opt": t_opt, "total": total})
    print("Step %2d: data=%.3f fwd=%.3f loss=%.3f bwd=%.3f opt=%.3f total=%.3f | data=%2.0f%% fwd=%2.0f%% loss=%2.0f%% bwd=%2.0f%% opt=%2.0f%%" %
          (step, t_data, t_fwd, t_loss, t_bwd, t_opt, total,
           t_data/total*100, t_fwd/total*100, t_loss/total*100, t_bwd/total*100, t_opt/total*100))

# Summary (skip first 3 warmup steps)
skip = 3
recs = records[skip:]
if recs:
    avg = {k: sum(r[k] for r in recs) / len(recs) for k in recs[0]}
    t = avg["total"]
    print("")
    print("=" * 70)
    print("PROFILING SUMMARY (avg of %d steps, excl %d warmup)" % (len(recs), skip))
    print("=" * 70)
    print("Data transfer:   %.4fs  (%5.1f%%)" % (avg["data"], avg["data"]/t*100))
    print("Forward pass:    %.4fs  (%5.1f%%)" % (avg["fwd"],  avg["fwd"]/t*100))
    print("Loss compute:    %.4fs  (%5.1f%%)" % (avg["loss"], avg["loss"]/t*100))
    print("Backward pass:   %.4fs  (%5.1f%%)" % (avg["bwd"],  avg["bwd"]/t*100))
    print("Optimizer step:  %.4fs  (%5.1f%%)" % (avg["opt"],  avg["opt"]/t*100))
    print("-" * 70)
    print("Total per step:  %.4fs" % t)
    print("Throughput:      %.0f samples/sec (single GPU)" % (256 / t))
    print("=" * 70)
