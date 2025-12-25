"""
U-Net Training Module (Zarr-based).

This module handles the training of a U-Net model for tumor segmentation
using pre-extracted Zarr patches. It implements:
- Dice + BCEWithLogitsLoss with class balancing
- Mixed precision training (AMP)
- Optimized DataLoader for Linux/Windows
- TensorBoard logging and checkpointing
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..models.unet_model import UNet
from ..datasets.zarr_segmentation_dataset import ZarrSegmentationDataset
from ..utils.device import get_device, print_device_info
from ..utils.reproducibility import set_seed
from ..utils.io import save_checkpoint, save_metrics, save_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DiceBCEWithLogitsLoss(nn.Module):
    """
    Combined Dice and BCEWithLogitsLoss for binary segmentation.
    
    Handles class imbalance via pos_weight in BCE and smooth Dice loss.
    """
    def __init__(self, alpha: float = 0.5, pos_weight: Optional[torch.Tensor] = None, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_logits: [B, C, H, W] (C=1 or 2)
            target: [B, 1, H, W] or [B, H, W]
        """
        # Ensure target is [B, 1, H, W]
        if target.dim() == 3:
            target = target.unsqueeze(1)
        
        # Extract tumor logits (assume class 1 if 2 channels, else class 0)
        if pred_logits.shape[1] == 2:
            tumor_logits = pred_logits[:, 1:2]
        else:
            tumor_logits = pred_logits
            
        # BCE Loss
        bce_loss = self.bce(tumor_logits, target.float())
        
        # Dice Loss
        pred_probs = torch.sigmoid(tumor_logits)
        intersection = (pred_probs * target.float()).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (
            pred_probs.sum() + target.float().sum() + self.smooth
        )
        
        return self.alpha * dice_loss + (1 - self.alpha) * bce_loss


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: GradScaler,
    amp_enabled: bool
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast('cuda', enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, masks)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        
    return total_loss / len(dataloader)


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool
) -> Tuple[float, float]:
    """Validate model and return (loss, iou)."""
    model.eval()
    total_loss = 0.0
    intersection = 0.0
    union = 0.0
    
    for images, masks in tqdm(dataloader, desc="Validation", leave=False):
        images, masks = images.to(device), masks.to(device)
        
        with autocast('cuda', enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, masks)
            
        total_loss += loss.item()
        
        # IoU calculation
        if logits.shape[1] == 2:
            preds = torch.sigmoid(logits[:, 1:2]) > 0.5
        else:
            preds = torch.sigmoid(logits) > 0.5
            
        intersection += (preds & (masks > 0)).sum().item()
        union += (preds | (masks > 0)).sum().item()
        
    iou = intersection / (union + 1e-6)
    return total_loss / len(dataloader), iou


def run_training_unet(
    zarr_path: str,
    output_dir: str = 'outputs/unet',
    train_split: float = 0.8,
    config: Optional[Dict] = None
):
    """
    Run U-Net training using Zarr dataset.
    """
    config = config or {}
    cfg = {
        'seed': config.get('seed', 42),
        'batch_size': config.get('batch_size', 32),
        'num_epochs': config.get('num_epochs', 50),
        'lr': config.get('lr', 1e-3),
        'weight_decay': config.get('weight_decay', 1e-4),
        'num_workers': config.get('num_workers', min(8, os.cpu_count() - 2) if os.name == 'posix' else 0),
        'amp_enabled': config.get('amp_enabled', True),
        'early_stop_patience': config.get('early_stop_patience', 10),
        # Model params
        'in_channels': 3,
        'num_classes': 2,
        'base_channels': 64,
    }
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Logging
    tb_writer = SummaryWriter(str(output_path / 'tensorboard_logs'))
    set_seed(cfg['seed'])
    device = get_device()
    print_device_info(device)
    save_config(cfg, output_path / 'config.yaml')
    
    # Dataset
    logger.info(f"Loading dataset from {zarr_path}")
    dataset = ZarrSegmentationDataset(zarr_path)
    
    # Split
    n_train = int(len(dataset) * train_split)
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(cfg['seed']))
    
    # Calculate pos_weight
    logger.info("Computing class weights...")
    sample_indices = torch.randperm(len(train_ds))[:100]
    tumor_px = 0
    total_px = 0
    for idx in sample_indices:
        _, mask = train_ds[idx]
        tumor_px += mask.sum().item()
        total_px += mask.numel()
    
    pos_weight_val = (total_px - tumor_px) / max(1, tumor_px)
    pos_weight = torch.tensor([pos_weight_val], device=device)
    logger.info(f"Pos weight: {pos_weight_val:.2f} (Tumor: {tumor_px/total_px:.1%})")
    
    # DataLoaders
    loader_args = dict(
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        pin_memory=True,
        persistent_workers=(cfg['num_workers'] > 0),
        prefetch_factor=2 if cfg['num_workers'] > 0 else None
    )
    train_dl = DataLoader(train_ds, shuffle=True, **loader_args)
    val_dl = DataLoader(val_ds, shuffle=False, **loader_args)
    
    # Model & Optimizer
    model = UNet(cfg['in_channels'], cfg['num_classes'], cfg['base_channels']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scaler = GradScaler('cuda', enabled=cfg['amp_enabled'])
    criterion = DiceBCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Training Loop
    best_iou = 0.0
    patience = 0
    
    logger.info("Starting training...")
    for epoch in range(1, cfg['num_epochs'] + 1):
        train_loss = train_epoch(model, train_dl, optimizer, criterion, device, scaler, cfg['amp_enabled'])
        val_loss, val_iou = validate(model, val_dl, criterion, device, cfg['amp_enabled'])
        
        # Logging
        logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val IoU={val_iou:.4f}")
        tb_writer.add_scalar('Loss/train', train_loss, epoch)
        tb_writer.add_scalar('Loss/val', val_loss, epoch)
        tb_writer.add_scalar('Metric/IoU', val_iou, epoch)
        
        # Checkpointing
        if val_iou > best_iou:
            best_iou = val_iou
            patience = 0
            save_checkpoint(model, optimizer, epoch, {'iou': val_iou}, output_path / 'best_model.pth', cfg)
            logger.info("✓ Saved best model")
        else:
            patience += 1
            if patience >= cfg['early_stop_patience']:
                logger.info("Early stopping triggered")
                break
                
        save_checkpoint(model, optimizer, epoch, {'iou': val_iou}, output_path / 'last_model.pth', cfg)

    tb_writer.close()
    logger.info(f"Training complete. Best IoU: {best_iou:.4f}")
