"""CLI helper to run a single patch PNG through a PyTorch .pth model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from inference.shared.patch_test_utils import (
    ACTIVATION_CHOICES,
    NORMALIZATION_CHOICES,
    PatchTestResult,
    build_result,
    load_class_names_file,
    load_pth_model,
    predict_patch,
    resolve_class_names,
)

SUPPORTED_ARCHES = ["resnet18", "resnet50"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Test a patch PNG against a .pth model")
    parser.add_argument("--model-path", required=True, help="Path to the .pth checkpoint")
    parser.add_argument("--arch", default="resnet50", choices=SUPPORTED_ARCHES, help="Backbone architecture")
    parser.add_argument("--patch-path", required=True, help="Path to the patch PNG/JPG/TIFF image")
    parser.add_argument("--image-size", type=int, default=224, help="Resize square dimension before inference")
    parser.add_argument(
        "--normalization", default="imagenet", choices=NORMALIZATION_CHOICES, help="Input normalization preset"
    )
    parser.add_argument(
        "--activation", default="auto", choices=ACTIVATION_CHOICES, help="Post-processing applied to logits"
    )
    parser.add_argument(
        "--class-names",
        default=None,
        help="Optional JSON/txt file describing class labels (overrides --num-classes)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Number of model outputs (required if --class-names is omitted)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="How many ranked predictions to print")
    parser.add_argument("--output-json", default=None, help="Optional path to serialize results")
    parser.add_argument("--device", default=None, help="Override torch.device string (e.g. 'cuda:0' or 'cpu')")
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Initialize backbone with torchvision pretrained weights before loading checkpoint",
    )
    return parser.parse_args()


def read_class_names(path: Optional[str], requested_classes: Optional[int]) -> List[str]:
    if path is not None:
        names = load_class_names_file(Path(path))
        return resolve_class_names(names, len(names))
    if not requested_classes:
        raise SystemExit("Either --class-names or --num-classes must be provided")
    return resolve_class_names(None, requested_classes)


def print_table(result: PatchTestResult) -> None:
    print("\nPer-class probabilities:")
    width = max(len(name) for name in result.class_names)
    for name, score in zip(result.class_names, result.probabilities):
        print(f"  {name:<{width}} : {score:0.6f}")

    print("\nTop predictions:")
    for row in result.top_k:
        print(f"  #{row['rank']} {row['class']} — {row['score']:0.6f}")


def main() -> None:
    args = parse_args()
    class_names = read_class_names(args.class_names, args.num_classes)
    num_outputs = len(class_names)

    model = load_pth_model(
        args.arch,
        num_outputs,
        Path(args.model_path),
        device=args.device,
        pretrained=args.pretrained,
    )

    probs = predict_patch(
        model,
        Path(args.patch_path),
        image_size=args.image_size,
        normalization=args.normalization,
        activation=args.activation,
        device=args.device,
    )

    probs_vector = np.array(probs).flatten()
    result = build_result(probs_vector, class_names, args.top_k)
    print_table(result)

    if args.output_json:
        payload = {
            "model_path": str(Path(args.model_path).resolve()),
            "patch_path": str(Path(args.patch_path).resolve()),
            "arch": args.arch,
            "image_size": args.image_size,
            "normalization": args.normalization,
            "activation": args.activation,
            "probabilities": result.probabilities,
            "class_names": result.class_names,
            "top_k": result.top_k,
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nJSON saved → {args.output_json}")


if __name__ == "__main__":
    main()
