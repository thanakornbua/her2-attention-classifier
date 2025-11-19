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
    p.add_argument("--model-path", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet50"])
    p.add_argument("--image-size", default=224, type=int)
    p.add_argument("--batch-size", default=None, type=int)
    p.add_argument("--num-workers", default=None, type=int)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--class-names-json", default=None)
    p.add_argument("--groups-csv", default=None)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--log-interval", default=50, type=int)
    return p.parse_args()


def build_dataloader(data_dir: Path, image_size: int, batch_size: int, num_workers: int) -> Tuple[DataLoader, List[str]]:
    if batch_size is None: batch_size = auto_scale_batch_size(image_size)
    if num_workers is None: num_workers = auto_detect_workers(batch_size)
    
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
            if json.load(open(checkpoint_path)).get('completed'): return
        except: pass
    
    dataloader, inferred_classes = build_dataloader(data_dir, args.image_size, args.batch_size, args.num_workers)
    class_names = json.load(open(args.class_names_json)) if args.class_names_json else inferred_classes
    print(f"Device: {device}, Classes: {len(class_names)}, Batch: {dataloader.batch_size}")
    
    model = load_classification_model(args.arch, len(class_names), Path(args.model_path), device)
    if hasattr(torch, 'compile') and torch.cuda.is_available():
        try: model = torch.compile(model, mode='reduce-overhead')
        except: pass
    
    num_samples = len(dataloader.dataset)
    y_true = np.zeros(num_samples, dtype=np.int64)
    y_pred_probs = np.zeros((num_samples, len(class_names)), dtype=np.float32)
    model.eval()
    
    if torch.cuda.is_available():
        with torch.no_grad(): _ = model(torch.randn(1, 3, args.image_size, args.image_size, device=device))
        torch.cuda.empty_cache()
    
    idx, oom_errors = 0, 0
    
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
                    print(f"{idx}/{num_samples} ({100*idx/num_samples:.0f}%)")
                if (batch_idx + 1) % 100 == 0:
                    try: json.dump({'processed': idx, 'total': num_samples}, open(checkpoint_path, 'w'))
                    except: pass
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    oom_errors += 1
                    if oom_errors >= 3: raise
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    continue
                raise
    
    y_pred = y_pred_probs.argmax(axis=1)
    metrics = classification_metrics(y_true, y_pred, average="macro")
    metrics["auroc_macro"] = multiclass_auroc(y_true, y_pred_probs)
    cm, tp, fn, fp, tn = confusion_matrix_and_counts(y_true, y_pred)
    
    json.dump(metrics, open(output_dir / "metrics.json", "w"), indent=2)
    write_confusion_matrix_csv(cm, class_names, output_dir / "confusion_matrix.csv")
    write_counts_csv(tp, fn, fp, tn, class_names, output_dir / "tp_fn_fp_tn.csv")
    
    if args.groups_csv:
        groups = load_groups(Path(args.groups_csv), dataloader.dataset)
        if groups: json.dump(group_metrics(y_true, y_pred, groups), open(output_dir / "group_metrics.json", "w"), indent=2)
    
    try: json.dump({'completed': True}, open(checkpoint_path, 'w'))
    except: pass
    
    print(f"\nAcc: {metrics['accuracy']:.4f} | AUROC: {metrics['auroc_macro']:.4f} → {output_dir}")


if __name__ == "__main__":
    main()
