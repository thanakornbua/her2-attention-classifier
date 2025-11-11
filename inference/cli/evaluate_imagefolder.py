"""
Classification evaluation CLI: metrics (Acc/P/R/F1/AUROC), confusion matrix, TP/FN/FP/TN, fairness.
Usage: python -m inference.cli.evaluate_imagefolder --model-path model.pth --data-dir test/ --arch resnet18 --output-dir results/
"""
from __future__ import annotations

import argparse, json
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


def parse_args():
    p = argparse.ArgumentParser("Evaluate classification on ImageFolder")
    p.add_argument("--model-path", required=True, help="Path to .pth checkpoint")
    p.add_argument("--data-dir", required=True, help="ImageFolder root")
    p.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet50"], help="Architecture")
    p.add_argument("--image-size", default=224, type=int, help="Image size")
    p.add_argument("--batch-size", default=32, type=int)
    p.add_argument("--num-workers", default=4, type=int)
    p.add_argument("--output-dir", required=True, help="Output directory")
    p.add_argument("--class-names-json", default=None, help="JSON with class names")
    p.add_argument("--groups-csv", default=None, help="CSV [filename,group] for fairness")
    return p.parse_args()


def build_dataloader(data_dir: Path, image_size: int, batch_size: int, num_workers: int) -> Tuple[DataLoader, List[str]]:
    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    dataset = datasets.ImageFolder(str(data_dir), transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
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
    
    print(f"Device: {device}")
    dataloader, inferred_classes = build_dataloader(data_dir, args.image_size, args.batch_size, args.num_workers)
    class_names = json.load(open(args.class_names_json)) if args.class_names_json else inferred_classes
    print(f"Classes: {len(class_names)}")
    
    model = load_classification_model(args.arch, len(class_names), Path(args.model_path), device)
    
    y_true, y_pred_probs = [], []
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            probs = torch.softmax(model(images), dim=1).cpu().numpy()
            y_pred_probs.extend(probs)
            y_true.extend(labels.cpu().numpy())
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * args.batch_size} samples")
    
    y_true, y_pred_probs = np.asarray(y_true), np.asarray(y_pred_probs)
    y_pred = y_pred_probs.argmax(axis=1)
    print(f"Total: {len(y_true)} samples")
    
    metrics = classification_metrics(y_true, y_pred, average="macro")
    metrics["auroc_macro"] = multiclass_auroc(y_true, y_pred_probs)
    cm, tp, fn, fp, tn = confusion_matrix_and_counts(y_true, y_pred)
    
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    write_confusion_matrix_csv(cm, class_names, output_dir / "confusion_matrix.csv")
    write_counts_csv(tp, fn, fp, tn, class_names, output_dir / "tp_fn_fp_tn.csv")
    
    if args.groups_csv:
        groups = load_groups(Path(args.groups_csv), dataloader.dataset)
        if groups:
            group_stats = group_metrics(y_true, y_pred, groups)
            with open(output_dir / "group_metrics.json", "w") as f:
                json.dump(group_stats, f, indent=2)
            print(f"Max fairness gap: {group_stats['gap_max']['value']:.4f}")
    
    print(f"\n{'='*60}\nEVALUATION SUMMARY\n{'='*60}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}  |  Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}  |  F1:        {metrics['f1']:.4f}")
    print(f"AUROC:     {metrics['auroc_macro']:.4f}\n{'='*60}\nResults: {output_dir}")


if __name__ == "__main__":
    main()
