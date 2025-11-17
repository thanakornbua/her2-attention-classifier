"""
Performance & robustness CLI: latency/throughput/VRAM, corruptions (noise/blur/JPEG/contrast), TTA.
Usage: python -m inference.cli.evaluate_performance --model-path model.pth --data-dir test/ --arch resnet18 --output-dir results/ --profile --robustness
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from inference.shared.load_model import load_classification_model
from src.utils.metric import classification_metrics, multiclass_auroc
from src.utils.perf import profile_inference
from src.utils.robustness import evaluate_corruptions, tta_flip4_predict

IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def auto_scale_batch_size(image_size: int) -> int:
    if image_size <= 224: return 32
    elif image_size <= 512: return 16
    elif image_size <= 1024: return 8
    else: return 4

def auto_detect_workers(batch_size: int) -> int:
    return min(os.cpu_count() or 4, batch_size, 8)


def parse_args():
    p = argparse.ArgumentParser("Performance & robustness evaluation")
    p.add_argument("--model-path", required=True, help="Path to .pth")
    p.add_argument("--data-dir", required=True, help="ImageFolder root")
    p.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet50"])
    p.add_argument("--image-size", default=224, type=int)
    # FIX #4: Auto-scale batch size for high-res
    p.add_argument("--batch-size", default=None, type=int, help="Batch size (auto if None)")
    p.add_argument("--num-workers", default=None, type=int, help="Data workers (auto if None)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--profile", action="store_true", help="Enable perf profiling")
    p.add_argument("--robustness", action="store_true", help="Enable corruption eval")
    p.add_argument("--tta", action="store_true", help="Use flip-4 TTA")
    p.add_argument("--warmup-batches", default=5, type=int)
    p.add_argument("--measure-batches", default=50, type=int)
    p.add_argument("--amp", action="store_true", help="Use AMP for 2x speedup")
    return p.parse_args()


def build_dataloader(data_dir: Path, image_size: int, batch_size: int, num_workers: int) -> Tuple[DataLoader, List[str]]:
    if batch_size is None:
        batch_size = auto_scale_batch_size(image_size)
        print(f"Auto-scaled batch_size={batch_size} for image_size={image_size}")
    if num_workers is None:
        num_workers = auto_detect_workers(batch_size)
        print(f"Auto-detected num_workers={num_workers}")
    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    dataset = datasets.ImageFolder(str(data_dir), transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
                           persistent_workers=False, prefetch_factor=2 if num_workers > 0 else None,
                           multiprocessing_context='spawn' if num_workers > 0 else None)
    return dataloader, dataset.classes


def evaluate_baseline(model, dataloader, device, use_tta: bool = False, use_amp: bool = False) -> dict:
    # FIX #1: Pre-allocate arrays to prevent memory leak from unbounded list growth
    num_samples = len(dataloader.dataset)
    num_classes = len(dataloader.dataset.classes)
    y_true = np.zeros(num_samples, dtype=np.int64)
    y_pred = np.zeros(num_samples, dtype=np.int64)
    y_pred_probs = np.zeros((num_samples, num_classes), dtype=np.float32)
    
    idx = 0
    model.eval()
    
    # FIX #11: CUDA OOM recovery
    oom_errors = 0
    max_oom_retries = 3
    
    with torch.no_grad():
        for images, labels in dataloader:
            try:
                batch_size = len(labels)
                images = images.to(device, non_blocking=True)
                
                # FIX #6: Automatic Mixed Precision for 2x speedup + 50% memory reduction
                with torch.cuda.amp.autocast(enabled=use_amp):
                    probs = (tta_flip4_predict(model, images) if use_tta else torch.softmax(model(images), dim=1)).cpu().numpy()
                
                y_pred[idx:idx+batch_size] = probs.argmax(axis=1)
                y_pred_probs[idx:idx+batch_size] = probs
                y_true[idx:idx+batch_size] = labels.cpu().numpy()
                idx += batch_size
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    oom_errors += 1
                    print(f"WARNING: CUDA OOM (error #{oom_errors})")
                    if oom_errors >= max_oom_retries: raise
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    print("  Cleared cache, skipping batch"); continue
                else: raise
    
    metrics = classification_metrics(y_true, y_pred, average="macro")
    metrics["auroc_macro"] = multiclass_auroc(y_true, y_pred_probs)
    return metrics


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir, output_dir = Path(args.data_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = output_dir / ".performance_checkpoint.json"
    if checkpoint_path.exists():
        try:
            checkpoint_data = json.load(open(checkpoint_path))
            if checkpoint_data.get('completed', False):
                print(f"Evaluation already completed. Delete {checkpoint_path} to re-run."); return
        except: pass
    
    print(f"Device: {device}")
    print(f"AMP: {'enabled' if args.amp else 'disabled'}")
    dataloader, class_names = build_dataloader(data_dir, args.image_size, args.batch_size, args.num_workers)
    model = load_classification_model(args.arch, len(class_names), Path(args.model_path), device)
    
    if hasattr(torch, 'compile') and torch.cuda.is_available() and not args.profile:
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("Model compiled with torch.compile()")
        except: pass
    
    results = {}
    baseline = evaluate_baseline(model, dataloader, device, use_tta=args.tta, use_amp=args.amp)
    results["baseline"] = baseline
    tta_label = " (TTA)" if args.tta else ""
    print(f"Baseline Acc{tta_label}: {baseline['accuracy']:.4f} | AUROC: {baseline['auroc_macro']:.4f}")
    
    if args.profile:
        print("\nProfiling...")
        perf = profile_inference(model, dataloader, device, warmup_batches=args.warmup_batches, measure_batches=args.measure_batches)
        results["performance"] = perf
        print(f"  Latency: p50={perf['latency_ms_p50']:.2f}ms p95={perf['latency_ms_p95']:.2f}ms")
        print(f"  Throughput: {perf['throughput_images_per_s']:.2f} img/s | VRAM: {perf['vram_peak_mb']:.2f}MB")
    
    if args.robustness:
        print("\nRobustness...")
        robust = evaluate_corruptions(model, dataloader, device)
        results["robustness"] = robust
        print(f"  Clean: {robust['clean_acc']:.4f}")
        for c in ["gaussian_noise", "gaussian_blur", "jpeg_compression", "brightness_contrast"]:
            if c in robust: print(f"  {c}: {robust[c]:+.4f} ({abs(robust[c])*100:.1f}% {'↓' if robust[c] > 0 else '↑'})")
    
    json.dump(results, open(output_dir / "evaluation.json", "w"), indent=2)
    try: json.dump({'completed': True}, open(checkpoint_path, 'w'))
    except: pass
    
    print(f"\n{'='*60}\nEVALUATION COMPLETE\n{'='*60}\nSaved: {output_dir / 'evaluation.json'}")
    if "robustness" in results:
        max_delta = max(abs(results["robustness"].get(c, 0)) for c in ["gaussian_noise", "gaussian_blur", "jpeg_compression", "brightness_contrast"])
        status = "✓ ROBUST (<5%)" if max_delta < 0.05 else "⚠ MODERATE (5-10%)" if max_delta < 0.10 else "✗ SENSITIVE (>10%)"
        print(f"\nMEDICAL READINESS: {status} - Max degradation: {max_delta*100:.1f}%")


if __name__ == "__main__":
    main()
