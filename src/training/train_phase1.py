"""
Phase 1: Patch-level classification training.

This module orchestrates the complete training pipeline for patch-level HER2 classification.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import Dict, Optional
import json
from torch.utils.tensorboard import SummaryWriter

from ..models.patch_classifier import PatchClassifier
from ..dataloader.zarr_patch_dataset import ZarrPatchDataset
from ..evaluation.metrics import compute_binary_metrics, find_optimal_threshold
from ..utils.device import get_device, print_device_info
from ..utils.reproducibility import set_seed
from ..utils.io import save_checkpoint, save_metrics, save_config, load_checkpoint


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool = True
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: PyTorch model
        dataloader: Training dataloader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        amp_enabled: Whether to use automatic mixed precision
        
    Returns:
        Average training loss
    """
    model.train()
    scaler = GradScaler('cuda', enabled=amp_enabled)
    
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for imgs, labels, _ in pbar:
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast('cuda', enabled=amp_enabled):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        
        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool = True
) -> Dict[str, float]:
    """
    Validate model.
    
    Args:
        model: PyTorch model
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to validate on
        amp_enabled: Whether to use AMP
        
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels, _ in tqdm(dataloader, desc='Validation'):
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            with autocast('cuda', enabled=amp_enabled):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item() * imgs.size(0)
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Concatenate all batches
    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # Compute metrics
    metrics = compute_binary_metrics(all_labels, all_probs[:, 1])
    metrics['val_loss'] = total_loss / len(dataloader.dataset)
    
    return metrics


def run_training(
    zarr_path: str,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    output_dir: str = 'outputs/phase1',
    config: Optional[Dict] = None,
    zarr_path_secondary: Optional[str] = None,
    patch_metadata: Optional['pd.DataFrame'] = None,
    resume_from: Optional[str] = None
):
    """
    Run complete Phase 1 training pipeline.
    
    Args:
        zarr_path: Path to primary Zarr dataset (normalized patches)
        train_indices: Training indices
        val_indices: Validation indices
        output_dir: Directory to save outputs
        config: Training configuration dictionary
        zarr_path_secondary: Optional path to secondary Zarr (raw patches) for dual-dataset training
        patch_metadata: Optional DataFrame with 'source' and 'her2_status' columns for dual Zarr routing
    """
    # Default config
    if config is None:
        config = {}
    
    cfg = {
        'seed': config.get('seed', 42),
        'batch_size': config.get('batch_size', 4),
        'num_epochs': config.get('num_epochs', 50),
        'lr': config.get('lr', 1e-4),
        'backbone': config.get('backbone', 'resnet50'),
        'num_classes': config.get('num_classes', 2),
        'dropout': config.get('dropout', 0.5),
        'amp_enabled': config.get('amp_enabled', True),
        'num_workers': config.get('num_workers', 0),
        'prefetch_factor': config.get('prefetch_factor', None),
        'persistent_workers': config.get('persistent_workers', False),
        'pin_memory': config.get('pin_memory', False),
        'early_stop_patience': config.get('early_stop_patience', 10),
        'resume_from': config.get('resume_from', resume_from),
    }
    
    # Setup
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize TensorBoard
    tb_log_dir = output_path / 'tensorboard_logs'
    writer = SummaryWriter(str(tb_log_dir))
    print(f"TensorBoard logs will be saved to: {tb_log_dir}")
    
    set_seed(cfg['seed'])
    device = get_device()
    print_device_info(device)
    
    # Save config
    save_config(cfg, output_path / 'config_used.yaml')
    
    # Data
    print("\nLoading datasets...")
    
    # Detect explicit routing mode (new recommended way)
    use_explicit_routing = (patch_metadata is not None and 
                           'zarr_path' in patch_metadata.columns and 
                           'zarr_index' in patch_metadata.columns)
    
    if use_explicit_routing:
        print(f"Using EXPLICIT Zarr routing (each row specifies its Zarr file and index)")
        train_ds = ZarrPatchDataset(
            zarr_root=None,  # Not used in explicit mode
            indices=train_indices,
            patch_metadata=patch_metadata
        )
        val_ds = ZarrPatchDataset(
            zarr_root=None,  # Not used in explicit mode
            indices=val_indices,
            patch_metadata=patch_metadata
        )
    # Legacy dual-Zarr mode (deprecated)
    elif zarr_path_secondary is not None and patch_metadata is not None:
        print(f"Using LEGACY dual Zarr archives (normalized + raw)")
        print(f"⚠️  Consider upgrading to explicit routing mode")
        train_ds = ZarrPatchDataset(
            zarr_path, 
            indices=train_indices,
            zarr_root_secondary=zarr_path_secondary,
            patch_metadata=patch_metadata
        )
        val_ds = ZarrPatchDataset(
            zarr_path, 
            indices=val_indices,
            zarr_root_secondary=zarr_path_secondary,
            patch_metadata=patch_metadata
        )
    # Legacy single-Zarr mode
    else:
        if zarr_path is None:
            raise ValueError("zarr_path cannot be None in legacy mode. Use explicit routing instead.")
        print(f"Using single Zarr archive (legacy mode)")
        train_ds = ZarrPatchDataset(zarr_path, indices=train_indices)
        val_ds = ZarrPatchDataset(zarr_path, indices=val_indices)
    
    dl_common = {
        'batch_size': cfg['batch_size'],
        'num_workers': cfg['num_workers'],
        'pin_memory': cfg['pin_memory'],
        'persistent_workers': cfg['persistent_workers'],
    }
    if cfg['num_workers'] > 0 and cfg['prefetch_factor'] is not None:
        dl_common['prefetch_factor'] = cfg['prefetch_factor']

    train_dl = DataLoader(train_ds, shuffle=True, **dl_common)
    val_dl = DataLoader(val_ds, shuffle=False, **dl_common)
    
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")
    
    # Model
    print(f"\nBuilding model (backbone={cfg['backbone']})...")
    model = PatchClassifier(
        backbone_name=cfg['backbone'],
        num_classes=cfg['num_classes'],
        dropout=cfg['dropout'],
        pretrained=True
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    
    # Training loop
    print("\nStarting training...")
    best_auc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}

    start_epoch = 1
    resume_path = cfg.get('resume_from')
    if resume_path:
        ckpt_path = Path(resume_path)
        if ckpt_path.exists():
            print(f"Resuming from checkpoint: {ckpt_path}")
            ckpt = load_checkpoint(ckpt_path, model=model, optimizer=optimizer, device=device, weights_only=False)
            start_epoch = ckpt.get('epoch', 0) + 1
            if 'metrics' in ckpt and 'auc' in ckpt['metrics']:
                best_auc = ckpt['metrics']['auc']
            print(f"Resumed at epoch {start_epoch-1} | Best AUC so far: {best_auc:.4f}")
        else:
            print(f"Warning: resume_from path not found: {ckpt_path}. Starting fresh.")

    for epoch in range(start_epoch, cfg['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{cfg['num_epochs']}")
        
        # Train
        train_loss = train_epoch(
            model, train_dl, optimizer, criterion, device, cfg['amp_enabled']
        )
        
        # Validate
        val_metrics = validate(
            model, val_dl, criterion, device, cfg['amp_enabled']
        )
        
        # Log
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['val_loss'])
        history['val_auc'].append(val_metrics['auc'])
        
        # TensorBoard logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_metrics['val_loss'], epoch)
        writer.add_scalar('Metrics/val_auc', val_metrics['auc'], epoch)
        writer.add_scalar('Metrics/val_accuracy', val_metrics['accuracy'], epoch)
        writer.add_scalar('Metrics/val_precision', val_metrics['precision'], epoch)
        writer.add_scalar('Metrics/val_recall', val_metrics['recall'], epoch)
        writer.add_scalar('Metrics/val_f1', val_metrics['f1'], epoch)
        writer.flush()
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"Val AUC: {val_metrics['auc']:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        
        # Save best model
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            patience_counter = 0
            
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                output_path / 'best_model.pth', config=cfg
            )
            print(f"✓ New best model saved (AUC={best_auc:.4f})")
        else:
            patience_counter += 1
        
        # Save last model
        save_checkpoint(
            model, optimizer, epoch, val_metrics,
            output_path / 'last_model.pth', config=cfg
        )
        
        # Early stopping
        if patience_counter >= cfg['early_stop_patience']:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
    
    # Save final metrics
    final_metrics = {
        'best_auc': best_auc,
        'history': history,
        'config': cfg
    }
    save_metrics(final_metrics, output_path / 'metrics.json')
    
    # Close TensorBoard writer
    writer.add_hparams(cfg, {'hparams/best_auc': best_auc})
    writer.close()
    
    print(f"\n✓ Training complete!")
    print(f"  Best AUC: {best_auc:.4f}")
    print(f"  Models saved to: {output_path}")
    print(f"  TensorBoard logs saved to: {tb_log_dir}")
    print(f"  To view: tensorboard --logdir={tb_log_dir}")
