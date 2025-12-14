#!/usr/bin/env python3
"""
Evaluate trained models on zarr test set.
Usage: python evaluate_zarr_model.py --model-path best_resnet50.pth --test-manifest outputs/zarr_test_manifest.csv --output-dir outputs/eval
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Defer importing build_model and dataset class until after path adjustments


def parse_args():
    p = argparse.ArgumentParser("Evaluate model on zarr test set")
    p.add_argument("--model-path", required=True, help="Path to trained model checkpoint (.pth)")
    p.add_argument("--test-manifest", required=True, help="Path to zarr test manifest CSV")
    p.add_argument("--output-dir", required=True, help="Directory to save evaluation results")
    p.add_argument("--arch", default="resnet50", choices=["resnet18", "resnet50", "efficientnet_b0"], help="Model architecture")
    p.add_argument("--num-classes", default=2, type=int, help="Number of classes")
    p.add_argument("--batch-size", default=32, type=int, help="Batch size for evaluation")
    p.add_argument("--num-workers", default=8, type=int, help="Number of dataloader workers")
    p.add_argument("--max-patches-per-slide", default=None, type=int, help="Max patches per slide (None = all)")
    p.add_argument("--amp", action="store_true", help="Use automatic mixed precision")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to use")
    p.add_argument("--limit-slides", type=int, default=None, help="Limit number of slides from manifest (random sample)")
    p.add_argument("--limit-patches", type=int, default=None, help="Limit total number of patches after loading slides (random sample)")
    return p.parse_args()


def load_model(arch: str, num_classes: int, checkpoint_path: Path, device: torch.device, build_model_fn):
    """Load trained model from checkpoint."""
    model = build_model_fn(arch, num_classes, dropout_p=0.0)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    return model


def evaluate_model(model, dataloader, device, use_amp=False):
    """Evaluate model and return predictions and ground truth."""
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return all_preds, all_probs, all_labels


def compute_metrics(y_true, y_pred, y_probs):
    """Compute classification metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    
    # Compute AUROC
    try:
        if y_probs.shape[1] == 2:
            metrics["auroc"] = float(roc_auc_score(y_true, y_probs[:, 1]))
        else:
            metrics["auroc"] = float(roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro"))
    except ValueError:
        metrics["auroc"] = float("nan")

    # Backwards compatibility with previous scripts expecting 'auroc_macro'
    metrics["auroc_macro"] = metrics["auroc"]
    
    return metrics


def save_confusion_matrix(y_true, y_pred, class_names, output_path):
    """Save confusion matrix as CSV."""
    cm = confusion_matrix(y_true, y_pred)
    
    with open(output_path, 'w') as f:
        f.write("," + ",".join(class_names) + "\n")
        for i, row in enumerate(cm):
            f.write(f"{class_names[i]},{','.join(map(str, row.tolist()))}\n")
    
    return cm


def save_per_class_metrics(y_true, y_pred, class_names, output_path):
    """Save per-class TP/FN/FP/TN counts."""
    cm = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    tn = cm.sum() - (tp + fn + fp)
    
    with open(output_path, 'w') as f:
        f.write("Class,TP,FN,FP,TN\n")
        for i, name in enumerate(class_names):
            f.write(f"{name},{int(tp[i])},{int(fn[i])},{int(fp[i])},{int(tn[i])}\n")


def main():
    args = parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Zarr Test Set Evaluation")
    print("=" * 60)
    print(f"Model:         {args.model_path}")
    print(f"Architecture:  {args.arch}")
    print(f"Test manifest: {args.test_manifest}")
    print(f"Device:        {device}")
    print(f"Output dir:    {output_dir}")
    print()
    
    # Load test dataset
    print("Loading test dataset...")
    from torchvision import transforms
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Import dataset & model builder from training code
    train_src_path = Path(__file__).parent / "src" / "train"
    sys.path.insert(0, str(train_src_path))
    try:
        from train_phase1_zarr import SimpleZarrPatchDataset, build_model as build_model_fn
    except ImportError as e:
        raise ImportError(f"Could not import training components from {train_src_path}: {e}")

    # Optionally reduce slides before dataset creation
    manifest_path = Path(args.test_manifest)
    if args.limit_slides is not None:
        df_manifest = pd.read_csv(manifest_path)
        if args.limit_slides < len(df_manifest):
            df_manifest = df_manifest.sample(n=args.limit_slides, random_state=42)
            limited_manifest_path = output_dir / f"_limited_test_manifest_{args.limit_slides}.csv"
            df_manifest.to_csv(limited_manifest_path, index=False)
            manifest_path = limited_manifest_path
            print(f"→ Using {len(df_manifest)} slides (limited from original {len(pd.read_csv(args.test_manifest))})")
        else:
            print(f"→ limit-slides >= total slides; using all {len(df_manifest)}")

    test_ds = SimpleZarrPatchDataset(
        manifest_path,
        transform=test_transform,
        max_patches_per_slide=args.max_patches_per_slide,
        seed=42,
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    original_slide_count = len(pd.read_csv(args.test_manifest))
    current_slide_count = len(pd.read_csv(manifest_path))
    print(f"✓ Loaded {len(test_ds)} test patches from {current_slide_count} slides (original manifest had {original_slide_count})")

    # Optionally limit patches globally after dataset construction
    if args.limit_patches is not None and args.limit_patches < len(test_ds):
        import random
        random.seed(42)
        # SimpleZarrPatchDataset exposes 'indices' list; subsample it
        test_ds.indices = random.sample(test_ds.indices, args.limit_patches)
        print(f"→ Limiting to {args.limit_patches} patches (random sample)")
    elif args.limit_patches is not None:
        print(f"→ limit-patches >= total patches; using all {len(test_ds)}")
    print()
    
    # Load model
    print("Loading model...")
    model = load_model(args.arch, args.num_classes, Path(args.model_path), device, build_model_fn)
    print(f"✓ Loaded {args.arch} model from {args.model_path}")
    print()
    
    # Evaluate
    print("Running evaluation...")
    y_pred, y_probs, y_true = evaluate_model(model, test_loader, device, args.amp)
    print(f"✓ Evaluated {len(y_pred)} patches")
    print()
    
    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(y_true, y_pred, y_probs)
    
    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save confusion matrix
    class_names = [f"class_{i}" for i in range(args.num_classes)]
    cm_path = output_dir / "confusion_matrix.csv"
    cm = save_confusion_matrix(y_true, y_pred, class_names, cm_path)
    
    # Save per-class metrics
    counts_path = output_dir / "tp_fn_fp_tn.csv"
    save_per_class_metrics(y_true, y_pred, class_names, counts_path)
    
    print("=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"AUROC:     {metrics['auroc']:.4f}")
    print()
    print("Confusion Matrix:")
    print(cm)
    print()
    print(f"✓ Results saved to: {output_dir}")
    print(f"  - {metrics_path}")
    print(f"  - {cm_path}")
    print(f"  - {counts_path}")


if __name__ == "__main__":
    main()
