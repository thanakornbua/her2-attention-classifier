"""
Performance profiling: Latency (p50/p95), throughput, VRAM peak, time per slide.
CUDA sync for accurate timing. Warmup batches exclude cold-start overhead.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def profile_inference(
    model, dataloader, device,
    warmup_batches: int = 5,
    measure_batches: int = 50,
    slide_ids: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """
    Profile inference: latency_ms (p50/p95), throughput, VRAM peak, time per slide.
    
    Returns: {"latency_ms_p50": 12.3, "latency_ms_p95": 15.8, "throughput_images_per_s": 2600,
              "vram_peak_mb": 3456, "time_per_slide_s": NaN (if no slide_ids)}
    """
    model.eval() if hasattr(model, "eval") else None
    
    latencies, per_sample_times = [], []
    total_images, total_time, batch_idx = 0, 0.0, 0
    
    use_cuda = bool(torch and torch.cuda.is_available())
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    
    ctx = torch.no_grad() if torch else _DummyContext()
    with ctx:
        for batch in dataloader:
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            if torch and isinstance(images, torch.Tensor):
                images = images.to(device, non_blocking=True)
            
            start = time.perf_counter()
            _ = model(images)
            if use_cuda:
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            
            if batch_idx >= warmup_batches:
                latencies.append(elapsed)
                bs = images.shape[0] if torch and isinstance(images, torch.Tensor) else len(images)
                total_images += bs
                total_time += elapsed
                per_sample_times.extend([elapsed / bs] * bs)
            
            batch_idx += 1
            if len(latencies) >= measure_batches:
                break
    
    # Compute metrics
    latency_ms = np.array(latencies) * 1000.0 if latencies else np.array([])
    p50 = float(np.percentile(latency_ms, 50)) if len(latency_ms) else float("nan")
    p95 = float(np.percentile(latency_ms, 95)) if len(latency_ms) else float("nan")
    throughput = float(total_images / total_time) if total_time > 0 else float("nan")
    vram_peak_mb = float(torch.cuda.max_memory_allocated(device) / 1024**2) if use_cuda else float("nan")
    
    # Slide-level timing
    time_per_slide_s = float("nan")
    if slide_ids and len(per_sample_times) == len(slide_ids):
        times_by_slide = {}
        for sid, t in zip(slide_ids, per_sample_times):
            times_by_slide.setdefault(sid, []).append(t)
        slide_times = [sum(times) for times in times_by_slide.values()]
        time_per_slide_s = float(np.mean(slide_times)) if slide_times else float("nan")
    
    return {
        "latency_ms_p50": p50,
        "latency_ms_p95": p95,
        "throughput_images_per_s": throughput,
        "vram_peak_mb": vram_peak_mb,
        "time_per_slide_s": time_per_slide_s,
    }


class _DummyContext:
    def __enter__(self): return self
    def __exit__(self, *_): return False
