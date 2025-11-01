import argparse
import os
import random
import shutil
import sys

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Randomly copy N folders (default 5, or fewer if not available) from a source directory
to a destination directory via CLI.

Usage:
    python random_copy.py /path/to/src /path/to/dst --num 5 --seed 42 --overwrite
"""


def list_subdirs(path, include_hidden=False):
    with os.scandir(path) as it:
        return [
            entry.name
            for entry in it
            if entry.is_dir() and (include_hidden or not entry.name.startswith("."))
        ]

def unique_dest_path(base_path):
    if not os.path.exists(base_path):
        return base_path
    parent, name = os.path.split(base_path)
    i = 1
    while True:
        candidate = os.path.join(parent, f"{name}_copy{i}")
        if not os.path.exists(candidate):
            return candidate
        i += 1

def copy_folder(src_folder, dst_folder, overwrite=False, dry_run=False):
    if dry_run:
        print(f"[DRY-RUN] Would copy: {src_folder} -> {dst_folder}")
        return
    if os.path.exists(dst_folder):
        if overwrite:
            shutil.rmtree(dst_folder)
        else:
            dst_folder = unique_dest_path(dst_folder)
    shutil.copytree(src_folder, dst_folder)
    print(f"Copied: {src_folder} -> {dst_folder}")

def main():
    p = argparse.ArgumentParser(description="Randomly copy folders from source to destination.")
    p.add_argument("src", help="Source directory containing folders")
    p.add_argument("dst", help="Destination directory where folders will be copied")
    p.add_argument("-n", "--num", type=int, default=5, help="Number of folders to copy (default: 5)")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing destination folders")
    p.add_argument("--dry-run", action="store_true", help="Show what would be copied without performing it")
    p.add_argument("--include-hidden", action="store_true", help="Include hidden folders (those starting with .)")
    args = p.parse_args()

    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)

    if not os.path.isdir(src):
        print(f"Source directory not found or not a directory: {src}", file=sys.stderr)
        sys.exit(2)

    os.makedirs(dst, exist_ok=True)

    subdirs = list_subdirs(src, include_hidden=args.include_hidden)
    if not subdirs:
        print("No folders found in source directory to copy.", file=sys.stderr)
        sys.exit(0)

    rng = random.Random(args.seed)
    count = min(args.num, len(subdirs))
    chosen = rng.sample(subdirs, count) if count < len(subdirs) else rng.sample(subdirs, len(subdirs))

    print(f"Selected {len(chosen)} folder(s) to copy from '{src}' to '{dst}':")
    for name in chosen:
        print(" -", name)

    for name in chosen:
        src_folder = os.path.join(src, name)
        dst_folder = os.path.join(dst, name)
        try:
            copy_folder(src_folder, dst_folder, overwrite=args.overwrite, dry_run=args.dry_run)
        except Exception as e:
            print(f"Failed to copy {src_folder} -> {dst_folder}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()