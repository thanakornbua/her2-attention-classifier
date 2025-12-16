"""
U-Net segmentation model training.

Trains U-Net for tumor segmentation using Dice + BCE loss.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional
import json
from torch.utils.tensorboard import SummaryWriter

from ..models.unet_model import UNet
from ..datasets.segmentation_dataset import SegmentationDataset
from ..utils.device import get_device, print_device_info
from ..utils.reproducibility import set_seed
from ..utils.io import save_checkpoint, save_metrics, save_config


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Dice loss for segmentation.
    
    Args:
        pred: Prediction logits [B, C, H, W]
        target: Ground truth [B, 1, H, W]
        smooth: Smoothing constant
        
    Returns:
        Dice loss
    """
    pred_probs = torch.softmax(pred, dim=1)
    pred_target = pred_probs[:, 1]  # Take tumor class
    
    intersection = (pred_target * target.float()).sum()
    dice = 1 - (2 * intersection + smooth) / (
        pred_target.sum() + target.float().sum() + smooth
    )
    
    return dice


def combined_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """
    Combined Dice + BCE loss.
    
    Args:
        pred: Prediction logits [B, C, H, W]
        target: Ground truth [B, 1, H, W]
        alpha: Weight for Dice loss
        
    Returns:
        Combined loss
    """
    # Dice loss
    dice = dice_loss(pred, target)
    
    # BCE loss
    pred_probs = torch.softmax(pred, dim=1)
    bce = nn.BCELoss()(pred_probs[:, 1], target.float().squeeze(1))
    
    return alpha * dice + (1 - alpha) * bce


def train_epoch_unet(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    amp_enabled: bool = True
) -> float:
    """
    Train U-Net for one epoch.
    
    Args:
        model: U-Net model
        dataloader: Training dataloader
        optimizer: Optimizer
        device: Device
        amp_enabled: Whether to use AMP
        
    Returns:
        Average training loss
    """
    model.train()
    scaler = GradScaler('cuda', enabled=amp_enabled)
    
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc='Training U-Net')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        with autocast('cuda', enabled=amp_enabled):
            logits = model(images)
            loss = combined_loss(logits, masks)
        
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


def validate_unet(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    amp_enabled: bool = True
) -> Dict[str, float]:
    """
    Validate U-Net model.
    
    Args:
        model: U-Net model
        dataloader: Validation dataloader
        device: Device
        amp_enabled: Whether to use AMP
        
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc='Validation'):
            images = images.to(device)
            masks = masks.to(device)
            
            with autocast('cuda', enabled=amp_enabled):
                logits = model(images)
                loss = combined_loss(logits, masks)
            
            total_loss += loss.item()
            
            preds = torch.softmax(logits, dim=1)[:, 1] > 0.5
            all_preds.append(preds.cpu().numpy())
            all_targets.append(masks.cpu().numpy())
    
    # Compute IoU
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    intersection = (all_preds * all_targets).sum()
    union = (all_preds + all_targets).sum() - intersection
    iou = intersection / (union + 1e-6)
    
    return {
        'val_loss': total_loss / len(dataloader),
        'iou': iou
    }


def run_training_unet(
    images_dir: str,
    masks_dir: str,
    train_split: float = 0.8,
    output_dir: str = 'outputs/unet',
    config: Optional[Dict] = None
):
    """
    Run complete U-Net training pipeline.
    
    Args:
        images_dir: Directory with images
        masks_dir: Directory with masks
        train_split: Train/val split ratio
        output_dir: Output directory
        config: Configuration dictionary
    """
    # Default config
    if config is None:
        config = {}
    
    cfg = {
        'seed': config.get('seed', 42),
        'batch_size': config.get('batch_size', 8),
        'num_epochs': config.get('num_epochs', 50),
        'lr': config.get('lr', 1e-4),
        'in_channels': config.get('in_channels', 3),
        'num_classes': config.get('num_classes', 2),
        'base_channels': config.get('base_channels', 64),
        'amp_enabled': config.get('amp_enabled', True),
        'num_workers': config.get('num_workers', 2),
        'early_stop_patience': config.get('early_stop_patience', 10),
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
    print("\nLoading segmentation dataset...")
    dataset = SegmentationDataset(images_dir, masks_dir)
    
    # Split
    n_train = int(train_split * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(
        dataset,
        [n_train, len(dataset) - n_train],
        generator=torch.Generator().manual_seed(cfg['seed'])
    )
    
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        pin_memory=True
    )
    
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=cfg['num_workers'],
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")
    
    # Model
    print("\nBuilding U-Net model...")
    model = UNet(
        in_channels=cfg['in_channels'],
        num_classes=cfg['num_classes'],
        base_channels=cfg['base_channels']
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=1e-5)
    
    # Training loop
    print("\nStarting U-Net training...")
    best_iou = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_iou': []}
    
    for epoch in range(1, cfg['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{cfg['num_epochs']}")
        
        # Train
        train_loss = train_epoch_unet(
            model, train_dl, optimizer, device, cfg['amp_enabled']
        )
        
        # Validate
        val_metrics = validate_unet(
            model, val_dl, device, cfg['amp_enabled']
        )
        
        # Log
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['val_loss'])
        history['val_iou'].append(val_metrics['iou'])
        
        # TensorBoard logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_metrics['val_loss'], epoch)
        writer.add_scalar('Metrics/val_iou', val_metrics['iou'], epoch)
        writer.add_scalar('Metrics/val_dice', val_metrics.get('dice', 0), epoch)
        writer.flush()
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"Val IoU: {val_metrics['iou']:.4f}")
        
        # Save best model
        if val_metrics['iou'] > best_iou:
            best_iou = val_metrics['iou']
            patience_counter = 0
            
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                output_path / 'best_model.pth', config=cfg
            )
            print(f"✓ New best model saved (IoU={best_iou:.4f})")
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
        'best_iou': best_iou,
        'history': history,
        'config': cfg
    }
    save_metrics(final_metrics, output_path / 'metrics.json')
    
    # Close TensorBoard writer
    writer.add_hparams(cfg, {'hparams/best_iou': best_iou})
    writer.close()
    
    print(f"\n✓ U-Net training complete!")
    print(f"  Best IoU: {best_iou:.4f}")
    print(f"  Models saved to: {output_path}")
    print(f"  TensorBoard logs saved to: {tb_log_dir}")
    print(f"  To view: tensorboard --logdir={tb_log_dir}")
