"""
Classification evaluation CLI: metrics (Acc/P/R/F1/AUROC), confusion matrix, TP/FN/FP/TN, fairness.
Usage: python -m inference.cli.evaluate_imagefolder --model-path model.pth --data-dir test/ --arch resnet18 --output-dir results/
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from inference.shared.load_model import load_classification_model
from src.utils.metric import classification_metrics, confusion_matrix_and_counts, group_metrics, multiclass_auroc, write_confusion_matrix_csv, write_counts_csv

IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def auto_scale_batch_size(image_size: int) -> int:
    """Auto-scale batch size based on resolution."""
    if image_size <= 224: return 32
    elif image_size <= 512: return 16
    elif image_size <= 1024: return 8
    else: return 4

def auto_detect_workers(batch_size: int) -> int:
    """Auto-detect optimal num_workers."""
    return min(os.cpu_count() or 4, batch_size, 8)


def parse_args():
    p = argparse.ArgumentParser("Evaluate classification on ImageFolder")
    p.add_argument("--model-path", required=True, help="Path to .pth checkpoint")
    p.add_argument("--data-dir", required=True, help="ImageFolder root")
    p.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet50"], help="Architecture")
    p.add_argument("--image-size", default=224, type=int, help="Image size")
    # FIX #4: Auto-scale batch size for high-res (default scales down automatically)
    p.add_argument("--batch-size", default=None, type=int, help="Batch size (auto if None)")
    p.add_argument("--num-workers", default=None, type=int, help="Data workers (auto if None)")
    p.add_argument("--output-dir", required=True, help="Output directory")
    p.add_argument("--class-names-json", default=None, help="JSON with class names")
    p.add_argument("--groups-csv", default=None, help="CSV [filename,group] for fairness")
    p.add_argument("--amp", action="store_true", help="Use Automatic Mixed Precision")
    p.add_argument("--log-interval", default=10, type=int, help="Log interval")
    return p.parse_args()


def build_dataloader(data_dir: Path, image_size: int, batch_size: int, num_workers: int) -> Tuple[DataLoader, List[str]]:
    if batch_size is None:
        batch_size = auto_scale_batch_size(image_size)
        print(f"Auto-scaled: batch_size={batch_size}, image_size={image_size}")
    if num_workers is None:
        num_workers = auto_detect_workers(batch_size)
        print(f"Auto-detected: num_workers={num_workers}")
    
    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    dataset = datasets.ImageFolder(str(data_dir), transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, 
                           pin_memory=True, persistent_workers=False, prefetch_factor=2 if num_workers > 0 else None,
                           multiprocessing_context='spawn' if num_workers > 0 else None)
    return dataloader, dataset.classes


def load_groups(groups_csv: Path, dataset: datasets.ImageFolder) -> Optional[List[str]]:
    try:
        df = pd.read_csv(groups_csv)
        if "filename" not in df.columns or "group" not in df.columns:
            print(f"Warning: {groups_csv} needs 'filename' and 'group' columns"); return None
        group_map = dict(zip(df["filename"], df["group"]))
        return [group_map.get(Path(path).name, "unknown") for path, _ in dataset.samples]
    except Exception as e:
        print(f"Warning: Could not load groups: {e}"); return None


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir, output_dir = Path(args.data_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = output_dir / ".evaluation_checkpoint.json"
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r') as f:
                if json.load(f).get('completed', False):
                    print(f"Already completed. Delete {checkpoint_path} to re-run.")
                    return
        except: pass
    
    print(f"Device: {device}")
    dataloader, inferred_classes = build_dataloader(data_dir, args.image_size, args.batch_size, args.num_workers)
    
    if args.class_names_json:
        try:
            with open(args.class_names_json, 'r') as f:
                class_names = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load class names ({e}), using inferred")
            class_names = inferred_classes
    else:
        class_names = inferred_classes
    print(f"Classes: {len(class_names)}")
    
    model = load_classification_model(args.arch, len(class_names), Path(args.model_path), device)
    
    # Compile model for 30% speedup (PyTorch 2.0+)
    if hasattr(torch, 'compile') and torch.cuda.is_available():
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("Model compiled")
        except: pass
    
    num_samples = len(dataloader.dataset)
    y_true = np.zeros(num_samples, dtype=np.int64)
    y_pred_probs = np.zeros((num_samples, len(class_names)), dtype=np.float32)
    model.eval()
    
    # Warmup
    if torch.cuda.is_available():
        with torch.no_grad():
            _ = model(torch.randn(1, 3, args.image_size, args.image_size, device=device))
        torch.cuda.empty_cache()
    
    idx, oom_errors, max_oom_retries = 0, 0, 3
    print(f"AMP: {'on' if args.amp else 'off'}, Samples: {num_samples}")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            try:
                images = images.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=args.amp):
                    probs = torch.softmax(model(images), dim=1).cpu().numpy()
                
                batch_size = probs.shape[0]
                y_pred_probs[idx:idx+batch_size] = probs
                y_true[idx:idx+batch_size] = labels.numpy()
                idx += batch_size
                
                if (batch_idx + 1) % args.log_interval == 0:
                    print(f"  {idx}/{num_samples} ({100*idx/num_samples:.1f}%)")
                
                if (batch_idx + 1) % 100 == 0:
                    try:
                        with open(checkpoint_path, 'w') as f:
                            json.dump({'processed_samples': idx, 'total_samples': num_samples, 'completed': False}, f)
                    except: pass
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    oom_errors += 1
                    if oom_errors >= max_oom_retries:
                        print(f"ERROR: Too many OOM errors, aborting at {idx}/{num_samples}")
                        raise
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    print(f"  OOM #{oom_errors}, skipping batch")
                    continue
                raise
    
    y_pred = y_pred_probs.argmax(axis=1)
    metrics = classification_metrics(y_true, y_pred, average="macro")
    metrics["auroc_macro"] = multiclass_auroc(y_true, y_pred_probs)
    cm, tp, fn, fp, tn = confusion_matrix_and_counts(y_true, y_pred)
    
    with open(output_dir / "metrics.json", "w") as f: json.dump(metrics, f, indent=2)
    write_confusion_matrix_csv(cm, class_names, output_dir / "confusion_matrix.csv")
    write_counts_csv(tp, fn, fp, tn, class_names, output_dir / "tp_fn_fp_tn.csv")
    
    if args.groups_csv:
        groups = load_groups(Path(args.groups_csv), dataloader.dataset)
        if groups:
            group_stats = group_metrics(y_true, y_pred, groups)
            with open(output_dir / "group_metrics.json", "w") as f: json.dump(group_stats, f, indent=2)
            print(f"Fairness gap: {group_stats['gap_max']['value']:.4f}")
    
    try:
        with open(checkpoint_path, 'w') as f: json.dump({'completed': True}, f)
    except: pass
    
    if torch.cuda.is_available():
        print(f"Peak GPU: {torch.cuda.max_memory_allocated(device)/(1024**2):.1f} MB")
    
    print(f"\n{'='*60}\nRESULTS\n{'='*60}")
    print(f"Acc: {metrics['accuracy']:.4f} | P: {metrics['precision']:.4f} | R: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f} | AUROC: {metrics['auroc_macro']:.4f}")
    print(f"{'='*60}\n{output_dir}")
    if oom_errors > 0: print(f"Warning: {oom_errors} OOM errors")


if __name__ == "__main__":
    main()
