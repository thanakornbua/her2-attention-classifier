"""Generate a class-name file from an ImageFolder directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from torchvision import datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Generate class-name file from ImageFolder-style data")
    parser.add_argument("--data-dir", required=True, help="Root directory arranged like torchvision.datasets.ImageFolder")
    parser.add_argument("--output", required=True, help="Destination file (.json or .txt)")
    parser.add_argument("--sort", action="store_true", help="Optional alphabetical sorting (default preserves folder order)")
    return parser.parse_args()


def write_output(names: list[str], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".json":
        with output.open("w", encoding="utf-8") as fh:
            json.dump(names, fh, indent=2)
    else:
        with output.open("w", encoding="utf-8") as fh:
            fh.write("\n".join(names) + "\n")


def main() -> None:
    args = parse_args()
    dataset = datasets.ImageFolder(args.data_dir)
    names = list(dataset.classes)
    if args.sort:
        names.sort()
    write_output(names, Path(args.output))
    print(f"Saved {len(names)} class names → {args.output}")


if __name__ == "__main__":
    main()
