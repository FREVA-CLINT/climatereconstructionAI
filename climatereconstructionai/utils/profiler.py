
import torch
from .. import config as cfg

def load_profiler(start_iter):
    if cfg.profile:
        return torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=cfg.max_iter - start_iter, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(cfg.log_dir),
        record_shapes=True)
    else:
        prof = lambda: None
        prof.start = lambda: None
        prof.step = lambda: None
        prof.stop = lambda: None
        return prof


