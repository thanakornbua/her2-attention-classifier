"""
Benchmarking script for Phase 1 training performance analysis.

Measures:
- Training time per epoch
- GPU utilization and memory usage
- Convergence speed (AUC progression)
- Throughput (samples/sec)

Usage:
    python train_phase1_benchmark.py --config configs/config.yaml
    
For distributed:
    torchrun --nproc_per_node=2 train_phase1_benchmark.py --config configs/config.yaml
"""

import argparse
import time
import json
import yaml
from pathlib import Path
import torch
import pandas as pd
import matplotlib.pyplot as plt
from train_phase1 import train_phase1

try:
    import pynvml  # Optional GPU utilization
    _HAS_PYNVML = True
except Exception:
    _HAS_PYNVML = False


def benchmark_training(config_path: str, output_dir: str = "outputs/benchmarks"):
    """Run training with detailed performance profiling."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override output directory for benchmark
    benchmark_dir = Path(output_dir)
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    config['output_dir'] = str(benchmark_dir / 'phase1_benchmark')
    
    # Collect GPU info before training
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            'gpu_name': torch.cuda.get_device_name(0),
            'gpu_count': torch.cuda.device_count(),
            'cuda_version': torch.version.cuda,
            'pytorch_version': torch.__version__,
        }
        print(f"GPU: {gpu_info['gpu_name']} (x{gpu_info['gpu_count']})")
    else:
        print("No GPU available - running on CPU")
    
    # Start benchmark
    print("\n" + "="*60)
    print("Starting Phase 1 Training Benchmark")
    print("="*60 + "\n")
    
    # Optional: initialize NVML for GPU utilization
    if _HAS_PYNVML and torch.cuda.is_available():
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    start_time = time.time()
    
    # Run training
    results = train_phase1(config)
    
    total_time = time.time() - start_time
    peak_mem_gb = None
    gpu_util = None
    if _HAS_PYNVML and torch.cuda.is_available():
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            peak_mem_gb = mem.used / 1024**3
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = util.gpu
        except Exception:
            pass
        finally:
            pynvml.nvmlShutdown()
    
    # Load training history
    logs_dir = Path(results['logs_dir'])
    metrics_df = pd.read_csv(logs_dir / 'metrics.csv')
    best_metrics = results['best_metrics']
    
    # Calculate benchmark metrics
    num_epochs = len(metrics_df)
    avg_time_per_epoch = total_time / num_epochs
    
    # Convergence analysis
    auc_progression = metrics_df['val_auc'].values
    epochs_to_90_auc = None
    epochs_to_95_auc = None
    for i, auc in enumerate(auc_progression):
        if auc >= 0.90 and epochs_to_90_auc is None:
            epochs_to_90_auc = i + 1
        if auc >= 0.95 and epochs_to_95_auc is None:
            epochs_to_95_auc = i + 1
    
    # Compile benchmark results
    benchmark_results = {
        'total_training_time_min': total_time / 60,
        'avg_time_per_epoch_sec': avg_time_per_epoch,
        'num_epochs': num_epochs,
        'best_val_auc': float(best_metrics['val']['auc']),
        'best_val_acc': float(best_metrics['val']['acc']),
        'best_epoch': best_metrics['epoch'],
        'epochs_to_90_auc': epochs_to_90_auc,
        'epochs_to_95_auc': epochs_to_95_auc,
        'final_val_auc': float(auc_progression[-1]),
        'gpu_info': gpu_info,
        'gpu_peak_mem_gb': peak_mem_gb,
        'gpu_util_percent': gpu_util,
        'config': config,
    }
    
    # Save benchmark results
    benchmark_json = benchmark_dir / 'benchmark_results.json'
    with open(benchmark_json, 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    
    print("\n" + "="*60)
    print("Benchmark Results")
    print("="*60)
    print(f"Total training time: {total_time / 60:.2f} minutes")
    print(f"Average time per epoch: {avg_time_per_epoch:.2f} seconds")
    print(f"Number of epochs: {num_epochs}")
    print(f"Best validation AUC: {benchmark_results['best_val_auc']:.4f} (epoch {benchmark_results['best_epoch']})")
    print(f"Best validation Accuracy: {benchmark_results['best_val_acc']:.4f}")
    print(f"Final validation AUC: {benchmark_results['final_val_auc']:.4f}")
    if benchmark_results['gpu_peak_mem_gb'] is not None:
        print(f"Peak GPU memory used: {benchmark_results['gpu_peak_mem_gb']:.2f} GB")
    if benchmark_results['gpu_util_percent'] is not None:
        print(f"Approx GPU utilization: {benchmark_results['gpu_util_percent']}%")
    if epochs_to_90_auc:
        print(f"Epochs to reach AUC ≥ 0.90: {epochs_to_90_auc}")
    else:
        print("Did not reach AUC ≥ 0.90")
    if epochs_to_95_auc:
        print(f"Epochs to reach AUC ≥ 0.95: {epochs_to_95_auc}")
    else:
        print("Did not reach AUC ≥ 0.95 (medical-grade target)")
    print(f"\nBenchmark results saved to: {benchmark_json}")
    
    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(metrics_df['epoch'], metrics_df['train_loss'], label='Train Loss', marker='o')
    axes[0, 0].plot(metrics_df['epoch'], metrics_df['val_loss'], label='Val Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # AUC curves
    axes[0, 1].plot(metrics_df['epoch'], metrics_df['train_auc'], label='Train AUC', marker='o')
    axes[0, 1].plot(metrics_df['epoch'], metrics_df['val_auc'], label='Val AUC', marker='s')
    axes[0, 1].axhline(y=0.95, color='r', linestyle='--', label='Medical-grade (0.95)', alpha=0.7)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('AUC')
    axes[0, 1].set_title('ROC-AUC Progression')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[1, 0].plot(metrics_df['epoch'], metrics_df['train_acc'], label='Train Acc', marker='o')
    axes[1, 0].plot(metrics_df['epoch'], metrics_df['val_acc'], label='Val Acc', marker='s')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Accuracy Progression')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Convergence analysis
    epochs_list = list(range(1, num_epochs + 1))
    axes[1, 1].plot(epochs_list, auc_progression, marker='o', linewidth=2, label='Val AUC')
    axes[1, 1].axhline(y=0.90, color='orange', linestyle='--', label='90% AUC', alpha=0.7)
    axes[1, 1].axhline(y=0.95, color='r', linestyle='--', label='95% AUC (Medical)', alpha=0.7)
    if epochs_to_90_auc:
        axes[1, 1].axvline(x=epochs_to_90_auc, color='orange', linestyle=':', alpha=0.5)
    if epochs_to_95_auc:
        axes[1, 1].axvline(x=epochs_to_95_auc, color='r', linestyle=':', alpha=0.5)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Validation AUC')
    axes[1, 1].set_title('Convergence Speed')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = benchmark_dir / 'training_curves.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {plot_path}")
    
    # Save a concise Markdown summary
    md = []
    md.append("# Phase 1 Benchmark Summary\n")
    md.append(f"- Total time: {total_time/60:.2f} min")
    md.append(f"- Avg time/epoch: {avg_time_per_epoch:.2f} s")
    md.append(f"- Best Val AUC: {benchmark_results['best_val_auc']:.4f} (epoch {benchmark_results['best_epoch']})")
    md.append(f"- Best Val Acc: {benchmark_results['best_val_acc']:.4f}")
    md.append(f"- Final Val AUC: {benchmark_results['final_val_auc']:.4f}")
    if epochs_to_90_auc: md.append(f"- Epochs to AUC≥0.90: {epochs_to_90_auc}")
    if epochs_to_95_auc: md.append(f"- Epochs to AUC≥0.95: {epochs_to_95_auc}")
    if peak_mem_gb is not None: md.append(f"- Peak GPU memory: {peak_mem_gb:.2f} GB")
    if gpu_util is not None: md.append(f"- Approx GPU utilization: {gpu_util}%")
    md.append("")
    md_path = benchmark_dir / 'benchmark_summary.md'
    md_path.write_text("\n".join(md), encoding='utf-8')
    print(f"Benchmark summary saved to: {md_path}")

    return benchmark_results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Phase 1 training performance")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--output', type=str, default='outputs/benchmarks', help='Output directory for benchmark results')
    args = parser.parse_args()
    
    benchmark_training(args.config, args.output)


if __name__ == '__main__':
    main()
