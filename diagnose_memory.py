#!/usr/bin/env python3
"""
Memory Leak Diagnostic Tool for HER2 Training Pipeline
Monitors memory usage and detects potential leaks during training.
"""

import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch


def get_gpu_memory():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        return allocated, reserved
    return 0, 0


def get_cpu_memory():
    """Get current process CPU memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2


def get_open_files():
    """Count number of open file descriptors."""
    process = psutil.Process(os.getpid())
    try:
        return len(process.open_files())
    except:
        return 0


def monitor_training(epochs=5, sleep_seconds=1):
    """
    Monitor memory during a brief training simulation.
    
    This helps identify if memory is being properly released between epochs.
    """
    print("=" * 60)
    print("Memory Leak Diagnostic Tool")
    print("=" * 60)
    print()
    
    print("Initial State:")
    print(f"  CPU Memory: {get_cpu_memory():.1f} MB")
    gpu_alloc, gpu_reserved = get_gpu_memory()
    print(f"  GPU Allocated: {gpu_alloc:.1f} MB")
    print(f"  GPU Reserved: {gpu_reserved:.1f} MB")
    print(f"  Open Files: {get_open_files()}")
    print()
    
    # Track memory over time
    memory_history = {
        'epoch': [],
        'cpu_mb': [],
        'gpu_alloc_mb': [],
        'gpu_reserved_mb': [],
        'open_files': []
    }
    
    print("Simulating training epochs...")
    print("-" * 60)
    
    for epoch in range(1, epochs + 1):
        # Simulate some tensor operations
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        # Create and delete some tensors
        for _ in range(10):
            x = torch.randn(100, 100, device=device)
            y = torch.randn(100, 100, device=device)
            z = torch.matmul(x, y)
            loss = z.mean()
            del x, y, z, loss
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        time.sleep(sleep_seconds)
        
        # Record metrics
        cpu_mem = get_cpu_memory()
        gpu_alloc, gpu_reserved = get_gpu_memory()
        open_files = get_open_files()
        
        memory_history['epoch'].append(epoch)
        memory_history['cpu_mb'].append(cpu_mem)
        memory_history['gpu_alloc_mb'].append(gpu_alloc)
        memory_history['gpu_reserved_mb'].append(gpu_reserved)
        memory_history['open_files'].append(open_files)
        
        print(f"Epoch {epoch:2d} | CPU: {cpu_mem:6.1f} MB | "
              f"GPU Alloc: {gpu_alloc:6.1f} MB | "
              f"GPU Rsrv: {gpu_reserved:6.1f} MB | "
              f"Files: {open_files:3d}")
    
    print("-" * 60)
    print()
    
    # Analysis
    print("Analysis:")
    cpu_growth = memory_history['cpu_mb'][-1] - memory_history['cpu_mb'][0]
    gpu_growth = memory_history['gpu_alloc_mb'][-1] - memory_history['gpu_alloc_mb'][0]
    files_growth = memory_history['open_files'][-1] - memory_history['open_files'][0]
    
    print(f"  CPU Memory Growth: {cpu_growth:+.1f} MB")
    print(f"  GPU Memory Growth: {gpu_growth:+.1f} MB")
    print(f"  Open Files Growth: {files_growth:+d}")
    print()
    
    # Verdict
    issues = []
    if cpu_growth > 100:
        issues.append(f"⚠ CPU memory growing ({cpu_growth:.1f} MB)")
    if gpu_growth > 100:
        issues.append(f"⚠ GPU memory growing ({gpu_growth:.1f} MB)")
    if files_growth > 10:
        issues.append(f"⚠ File descriptors growing ({files_growth:+d})")
    
    if issues:
        print("Potential Issues Detected:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✓ No significant memory leaks detected")
    
    print()
    return memory_history


def check_zarr_references():
    """Test if zarr array references are being properly released."""
    print("Testing Zarr Array Reference Handling...")
    print("-" * 60)
    
    # Create a temporary zarr array
    import zarr
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        z = zarr.open(f"{tmpdir}/test.zarr", mode='w', shape=(100, 100, 3), dtype='uint8')
        z[:] = np.random.randint(0, 255, (100, 100, 3), dtype='uint8')
        
        initial_mem = get_cpu_memory()
        
        # Test WITHOUT copy (bad)
        from PIL import Image
        refs_bad = []
        for i in range(10):
            arr = np.array(z[i])  # No copy
            img = Image.fromarray(arr)
            refs_bad.append(img)
        
        mem_no_copy = get_cpu_memory() - initial_mem
        print(f"  Without copy=True: +{mem_no_copy:.1f} MB")
        
        del refs_bad
        gc.collect()
        
        # Test WITH copy (good)
        refs_good = []
        for i in range(10):
            arr = np.array(z[i], copy=True)  # With copy
            img = Image.fromarray(arr)
            refs_good.append(img)
        
        mem_with_copy = get_cpu_memory() - initial_mem
        print(f"  With copy=True:    +{mem_with_copy:.1f} MB")
        
        del refs_good
        gc.collect()
    
    if mem_with_copy < mem_no_copy * 0.8:
        print("  ✓ copy=True reduces memory usage")
    else:
        print("  ⚠ No significant difference detected")
    
    print()


def main():
    """Run all diagnostic checks."""
    print()
    
    # Check zarr handling
    check_zarr_references()
    
    # Monitor simulated training
    monitor_training(epochs=10, sleep_seconds=0.5)
    
    print("=" * 60)
    print("Diagnostic Complete")
    print("=" * 60)
    print()
    print("To monitor real training:")
    print("  1. Run training in one terminal: ./train.sh")
    print("  2. Monitor GPU in another: watch -n 1 nvidia-smi")
    print("  3. Check this script's output for baseline expectations")
    print()


if __name__ == "__main__":
    main()
