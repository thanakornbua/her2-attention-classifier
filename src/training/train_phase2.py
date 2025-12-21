"""
Phase 2: Multiple Instance Learning (MIL) training for slide-level classification.

This module:
1. Loads frozen Phase 1 backbone
2. Extracts patch embeddings from all slides
3. Trains attention-based MIL model for slide-level prediction
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
from ..models.mil_model import AttentionMIL, GatedAttentionMIL
from ..datasets.mil_dataset import SlideFeatureDataset, collate_slide_features
from ..dataloader.zarr_patch_dataset import ZarrPatchDataset
from ..evaluation.metrics import compute_binary_metrics
from ..utils.device import get_device, print_device_info
from ..utils.reproducibility import set_seed
from ..utils.io import save_checkpoint, save_metrics, save_config, load_checkpoint


def warn_if_case_domain_mixed(
    patch_metadata: pd.DataFrame,
    *,
    case_col: str = "case_name",
    domain_col: str = "source",
    warn_threshold: int = 0,
) -> None:
    """Warn if any case mixes domains (raw/normalized).

    For MIL, mixed-domain bags can cause the attention head to latch onto stain domain.
    This helper is intentionally non-fatal.
    """
    if patch_metadata is None or patch_metadata.empty:
        return
    if case_col not in patch_metadata.columns or domain_col not in patch_metadata.columns:
        return

    mixed = patch_metadata.groupby(case_col, dropna=False)[domain_col].nunique(dropna=False)
    n_mixed = int((mixed > 1).sum())
    if n_mixed > warn_threshold:
        print(
            f"⚠️  MIL heads-up: {n_mixed} case(s) have mixed domains in patch_metadata "
            f"(column '{domain_col}'). Consider per-case domain consistency or domain conditioning."
        )


def extract_features(
    backbone: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    amp_enabled: bool = True
) -> Dict[str, np.ndarray]:
    """
    Extract features from patches using frozen backbone.
    
    Args:
        backbone: Feature extraction model
        dataloader: DataLoader with patches
        device: Device to use
        amp_enabled: Whether to use AMP
        
    Returns:
        Dictionary mapping slide_id -> features [num_patches, feature_dim]
    """
    backbone.eval()
    
    features_dict = {}
    
    with torch.no_grad():
        for imgs, labels, metadata in tqdm(dataloader, desc='Extracting features'):
            imgs = imgs.to(device)
            
            with autocast('cuda', enabled=amp_enabled):
                # Get features before classification head
                feats = backbone.backbone(imgs)  # [B, feat_dim, H, W]
                feats = backbone.gap(feats)      # [B, feat_dim, 1, 1]
                feats = feats.flatten(1)         # [B, feat_dim]
            
            feats_np = feats.cpu().numpy()
            
            # Group by slide_id (assuming metadata contains slide identifiers)
            for i, feat in enumerate(feats_np):
                slide_id = metadata[i] if isinstance(metadata, list) else f"slide_{i}"
                
                if slide_id not in features_dict:
                    features_dict[slide_id] = []
                
                features_dict[slide_id].append(feat)
    
    # Convert lists to arrays
    for slide_id in features_dict:
        features_dict[slide_id] = np.stack(features_dict[slide_id])
    
    return features_dict


def train_epoch_mil(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool = True
) -> float:
    """
    Train MIL model for one epoch.
    
    Args:
        model: MIL model
        dataloader: Slide-level dataloader
        optimizer: Optimizer
        criterion: Loss function
        device: Device
        amp_enabled: Whether to use AMP
        
    Returns:
        Average training loss
    """
    model.train()
    scaler = GradScaler('cuda', enabled=amp_enabled)
    
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc='Training MIL')
    for features, labels, slide_ids, lengths in pbar:
        features = features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast('cuda', enabled=amp_enabled):
            logits = model(features)
            loss = criterion(logits, labels)
        
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


def validate_mil(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool = True,
    save_attention: bool = False,
    output_dir: Optional[Path] = None
) -> Dict[str, float]:
    """
    Validate MIL model and optionally save attention weights.
    
    Args:
        model: MIL model
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device
        amp_enabled: Whether to use AMP
        save_attention: Whether to save attention weights
        output_dir: Directory to save attention weights
        
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    
    total_loss = 0.0
    all_probs = []
    all_preds = []
    all_labels = []
    attention_weights_dict = {}
    
    with torch.no_grad():
        for features, labels, slide_ids, lengths in tqdm(dataloader, desc='Validation'):
            features = features.to(device)
            labels = labels.to(device)
            
            with autocast('cuda', enabled=amp_enabled):
                if save_attention:
                    logits, attention = model(features, return_attention=True)
                    
                    # Save attention weights per slide
                    for i, slide_id in enumerate(slide_ids):
                        length = lengths[i].item()
                        attention_weights_dict[slide_id] = attention[i, :length].cpu().numpy()
                else:
                    logits = model(features)
                
                loss = criterion(logits, labels)
            
            total_loss += loss.item() * features.size(0)
            
            probs = torch.softmax(logits, dim=1)
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
    
    # Save attention weights
    if save_attention and output_dir is not None:
        attention_dir = output_dir / 'attention_weights'
        attention_dir.mkdir(parents=True, exist_ok=True)
        
        for slide_id, weights in attention_weights_dict.items():
            np.save(attention_dir / f"{slide_id}.npy", weights)
    
    return metrics, attention_weights_dict if save_attention else None


def run_training_phase2(
    features_dir: str,
    labels_csv: str,
    train_slide_ids: np.ndarray,
    val_slide_ids: np.ndarray,
    output_dir: str = 'outputs/phase2',
    config: Optional[Dict] = None
):
    """
    Run complete Phase 2 MIL training pipeline.
    
    Args:
        features_dir: Directory with pre-extracted features
        labels_csv: CSV with slide labels
        train_slide_ids: Training slide IDs
        val_slide_ids: Validation slide IDs
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
        'feature_dim': config.get('feature_dim', 2048),
        'hidden_dim': config.get('hidden_dim', 512),
        'num_classes': config.get('num_classes', 2),
        'dropout': config.get('dropout', 0.25),
        'model_type': config.get('model_type', 'attention'),  # 'attention' or 'gated'
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
    print("\nLoading slide datasets...")
    train_ds = SlideFeatureDataset(features_dir, labels_csv, train_slide_ids)
    val_ds = SlideFeatureDataset(features_dir, labels_csv, val_slide_ids)
    
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        collate_fn=collate_slide_features,
        pin_memory=True
    )
    
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=cfg['num_workers'],
        collate_fn=collate_slide_features,
        pin_memory=True
    )
    
    print(f"Train slides: {len(train_ds)}")
    print(f"Val slides: {len(val_ds)}")
    
    # Model
    print(f"\nBuilding MIL model (type={cfg['model_type']})...")
    if cfg['model_type'] == 'gated':
        model = GatedAttentionMIL(
            feature_dim=cfg['feature_dim'],
            hidden_dim=cfg['hidden_dim'],
            num_classes=cfg['num_classes'],
            dropout=cfg['dropout']
        ).to(device)
    else:
        model = AttentionMIL(
            feature_dim=cfg['feature_dim'],
            hidden_dim=cfg['hidden_dim'],
            num_classes=cfg['num_classes'],
            dropout=cfg['dropout']
        ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    
    # Training loop
    print("\nStarting MIL training...")
    best_auc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
    
    for epoch in range(1, cfg['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{cfg['num_epochs']}")
        
        # Train
        train_loss = train_epoch_mil(
            model, train_dl, optimizer, criterion, device, cfg['amp_enabled']
        )
        
        # Validate (save attention on last epoch)
        save_attn = (epoch == cfg['num_epochs'])
        val_metrics, attn_weights = validate_mil(
            model, val_dl, criterion, device, cfg['amp_enabled'],
            save_attention=save_attn, output_dir=output_path
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
    
    print(f"\n✓ MIL training complete!")
    print(f"  Best AUC: {best_auc:.4f}")
    print(f"  Models saved to: {output_path}")
    print(f"  TensorBoard logs saved to: {tb_log_dir}")
    print(f"  To view: tensorboard --logdir={tb_log_dir}")
    if attn_weights:
        print(f"  Attention weights saved to: {output_path / 'attention_weights'}")
