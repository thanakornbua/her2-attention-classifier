"""
Performance & robustness CLI: latency/throughput/VRAM, corruptions (noise/blur/JPEG/contrast), TTA.
Usage: python -m inference.cli.evaluate_performance --model-path model.pth --data-dir test/ --arch resnet18 --output-dir results/ --profile --robustness
"""
from __future__ import annotations

import argparse, json
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


def parse_args():
    p = argparse.ArgumentParser("Performance & robustness evaluation")
    p.add_argument("--model-path", required=True, help="Path to .pth")
    p.add_argument("--data-dir", required=True, help="ImageFolder root")
    p.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet50"])
    p.add_argument("--image-size", default=224, type=int)
    p.add_argument("--batch-size", default=32, type=int)
    p.add_argument("--num-workers", default=4, type=int)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--profile", action="store_true", help="Enable perf profiling")
    p.add_argument("--robustness", action="store_true", help="Enable corruption eval")
    p.add_argument("--tta", action="store_true", help="Use flip-4 TTA")
    p.add_argument("--warmup-batches", default=5, type=int)
    p.add_argument("--measure-batches", default=50, type=int)
    return p.parse_args()


def build_dataloader(data_dir: Path, image_size: int, batch_size: int, num_workers: int) -> Tuple[DataLoader, List[str]]:
    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    dataset = datasets.ImageFolder(str(data_dir), transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True), dataset.classes


def evaluate_baseline(model, dataloader, device, use_tta: bool = False) -> dict:
    y_true, y_pred, y_pred_probs = [], [], []
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            probs = (tta_flip4_predict(model, images) if use_tta else torch.softmax(model(images), dim=1)).cpu().numpy()
            y_pred.extend(probs.argmax(axis=1))
            y_pred_probs.extend(probs)
            y_true.extend(labels.cpu().numpy())
    y_true, y_pred, y_pred_probs = np.asarray(y_true), np.asarray(y_pred), np.asarray(y_pred_probs)
    metrics = classification_metrics(y_true, y_pred, average="macro")
    metrics["auroc_macro"] = multiclass_auroc(y_true, y_pred_probs)
    return metrics


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir, output_dir = Path(args.data_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Device: {device}")
    dataloader, class_names = build_dataloader(data_dir, args.image_size, args.batch_size, args.num_workers)
    model = load_classification_model(args.arch, len(class_names), Path(args.model_path), device)
    
    results = {}
    baseline = evaluate_baseline(model, dataloader, device, use_tta=args.tta)
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
        for corruption in ["gaussian_noise", "gaussian_blur", "jpeg_compression", "brightness_contrast"]:
            if corruption in robust:
                delta = robust[corruption]
                print(f"  {corruption}: {delta:+.4f} ({abs(delta)*100:.1f}% {'↓' if delta > 0 else '↑'})")
    
    with open(output_dir / "evaluation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*60}\nEVALUATION COMPLETE\n{'='*60}\nSaved: {output_dir / 'evaluation.json'}")
    
    if "robustness" in results:
        max_delta = max(abs(results["robustness"].get(c, 0)) for c in ["gaussian_noise", "gaussian_blur", "jpeg_compression", "brightness_contrast"])
        status = "✓ ROBUST (<5%)" if max_delta < 0.05 else "⚠ MODERATE (5-10%)" if max_delta < 0.10 else "✗ SENSITIVE (>10%)"
        print(f"\nMEDICAL READINESS: {status} - Max degradation: {max_delta*100:.1f}%")


if __name__ == "__main__":
    main()
