#!/usr/bin/env python3
"""Simple launcher for Phase 1 training with automatic DDP setup."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == '__main__':
    import argparse
    import yaml
    from src.train.train_phase1 import train_phase1
    
    parser = argparse.ArgumentParser(description='Phase 1 Training Launcher')
    parser.add_argument('--config', required=True, help='Config YAML file')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs (-1=all, 0=CPU)')
    parser.add_argument('--focal-loss', action='store_true', help='Use focal loss')
    args = parser.parse_args()
    
    # Load and run
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Auto-configure based on args
    if args.gpus != 1:
        import torch
        num_gpus = torch.cuda.device_count() if args.gpus == -1 else args.gpus
        config['use_ddp'] = (num_gpus > 1)
        print(f"Using {num_gpus} GPU(s), DDP={'ON' if config['use_ddp'] else 'OFF'}")
    
    if args.focal_loss:
        config['loss_type'] = 'focal'
    
    # Run training
    train_phase1(config)
