"""
Deprecated file: this benchmark script has been removed to keep the repository minimal.

Use the main training entrypoint instead:
  - Single GPU/CPU:    python -m src.train.train_phase1 --config configs/config.yaml
  - Multi-GPU (DDP):   torchrun --nproc_per_node=N -m src.train.train_phase1 --config configs/config.yaml

If you need performance profiling, integrate it directly into your run or a separate downstream script.
"""

import sys

def main() -> None:
    msg = (
        "This file is deprecated and no longer supported.\n"
        "Please run training via the module entrypoint instead:\n"
        "  python -m src.train.train_phase1 --config configs/config.yaml\n"
        "  torchrun --nproc_per_node=N -m src.train.train_phase1 --config configs/config.yaml\n"
    )
    print(msg)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
