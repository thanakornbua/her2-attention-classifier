"""
Phase 1 patch-level training entrypoint (ResNet-50) - OPTIMIZED VERSION

Fixes:
- Memory leaks from PIL images, numpy arrays, and gradient accumulation
- Performance monitoring with detailed metrics
- Efficient tensor operations and memory management
- Better logging and checkpoint management
"""

from __future__ import annotations

import os
import json
import time
import random
import psutil
import gc
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import contextmanager

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as T
import torchvision.models as models
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report

# Stain normalization (Macenko)
from src.preprocessing.stain_normalization import macenko_normalization

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")


# ============================================================================
# PERFORMANCE MONITORING UTILITIES
# ============================================================================

class PerformanceMonitor:
    """Track detailed performance metrics during training."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.reset()
    
    def reset(self):
        """Reset counters for new epoch."""
        self.batch_times = []
        self.data_loading_times = []
        self.forward_times = []
        self.backward_times = []
        self.optimizer_times = []
        self.start_time = time.time()
    
    def get_memory_stats(self):
        """Get current memory usage statistics."""
        stats = {}
        
        # CPU memory
        mem_info = self.process.memory_info()
        stats['cpu_ram_mb'] = mem_info.rss / 1024**2
        stats['cpu_ram_percent'] = self.process.memory_percent()
        
        # GPU memory
        if torch.cuda.is_available():
            stats['gpu_allocated_mb'] = torch.cuda.memory_allocated() / 1024**2
            stats['gpu_reserved_mb'] = torch.cuda.memory_reserved() / 1024**2
            stats['gpu_max_allocated_mb'] = torch.cuda.max_memory_allocated() / 1024**2
            stats['gpu_percent'] = (torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory) * 100
        
        return stats
    
    def get_throughput_stats(self, num_samples):
        """Calculate throughput statistics."""
        elapsed = time.time() - self.start_time
        return {
            'samples_per_sec': num_samples / elapsed if elapsed > 0 else 0,
            'avg_batch_time_ms': np.mean(self.batch_times) * 1000 if self.batch_times else 0,
            'avg_data_load_ms': np.mean(self.data_loading_times) * 1000 if self.data_loading_times else 0,
            'avg_forward_ms': np.mean(self.forward_times) * 1000 if self.forward_times else 0,
            'avg_backward_ms': np.mean(self.backward_times) * 1000 if self.backward_times else 0,
            'data_load_percent': (np.sum(self.data_loading_times) / elapsed * 100) if elapsed > 0 else 0,
        }
    
    @contextmanager
    def time_operation(self, operation_name):
        """Context manager to time operations."""
        start = time.time()
        yield
        elapsed = time.time() - start
        
        if operation_name == 'batch':
            self.batch_times.append(elapsed)
        elif operation_name == 'data_loading':
            self.data_loading_times.append(elapsed)
        elif operation_name == 'forward':
            self.forward_times.append(elapsed)
        elif operation_name == 'backward':
            self.backward_times.append(elapsed)
        elif operation_name == 'optimizer':
            self.optimizer_times.append(elapsed)


# ============================================================================
# MEMORY-EFFICIENT DATASET
# ============================================================================

class PatchDataset(Dataset):
    """Memory-efficient patch dataset with proper resource cleanup."""
    
    def __init__(self, csv_path: str, path_col: str, label_col: str, transform=None):
        self.df = pd.read_csv(csv_path)
        self.path_col = path_col
        self.label_col = label_col
        self.transform = transform
        
        # Validate columns
        if self.path_col not in self.df.columns:
            raise ValueError(f"Missing column '{self.path_col}' in {csv_path}")
        if self.label_col not in self.df.columns:
            if 'target' in self.df.columns:
                self.label_col = 'target'
            else:
                raise ValueError(f"Missing label column in {csv_path}")
        
        # Convert paths to strings and labels to int upfront
        self.df[self.path_col] = self.df[self.path_col].astype(str)
        self.df[self.label_col] = self.df[self.label_col].astype(int)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row[self.path_col]
        label = row[self.label_col]
        
        # FIXED: Use context manager to ensure proper cleanup
        try:
            with Image.open(path) as img:
                # Convert to RGB immediately
                img_rgb = img.convert('RGB')
                
                # Apply transforms if needed
                if self.transform:
                    img_tensor = self.transform(img_rgb)
                else:
                    img_tensor = T.ToTensor()(img_rgb)
            
            return img_tensor, label, path
        
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            # Return a black image as fallback to prevent training crash
            img_tensor = torch.zeros((3, 224, 224))
            return img_tensor, label, path


# ============================================================================
# OPTIMIZED TRANSFORMS WITH MEMORY MANAGEMENT
# ============================================================================

class MacenkoNormalizeTransform:
    """Memory-efficient Macenko normalization."""
    
    def __call__(self, img: Image.Image) -> Image.Image:
        # Convert to numpy array
        arr = np.array(img, dtype=np.uint8)
        
        try:
            # Apply normalization
            norm = macenko_normalization(arr)
            
            # Explicitly delete input array to free memory
            del arr
            
            # Convert back to PIL
            result = Image.fromarray(norm.astype(np.uint8))
            
            # Delete normalized array
            del norm
            
            return result
        
        except Exception as e:
            # Fallback: return original image
            del arr
            return img


def build_transforms(input_size: int, train: bool = True):
    """Build transforms with memory-efficient operations."""
    tfms = []
    
    # Add Macenko normalization
    tfms.append(MacenkoNormalizeTransform())
    
    # Resize
    tfms.append(T.Resize((input_size, input_size), antialias=True))
    
    # Augmentations for training
    if train:
        tfms.extend([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        ])
    
    # Convert to tensor and normalize
    tfms.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return T.Compose(tfms)


# ============================================================================
# MODEL BUILDING
# ============================================================================

def build_resnet50(pretrained: bool = True, num_classes: int = 2):
    """Build ResNet-50 model."""
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
    except Exception:
        model = models.resnet50(pretrained=pretrained)
    
    in_feat = model.fc.in_features
    model.fc = nn.Linear(in_feat, num_classes)
    
    return model, 'layer4'


def build_optimizer(model, lr=1e-4, weight_decay=1e-4):
    """Build optimizer."""
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


# ============================================================================
# MEMORY-EFFICIENT TRAINING LOOP
# ============================================================================

def train_one_epoch(
    model, 
    loader, 
    criterion, 
    optimizer, 
    device, 
    accumulation_steps=1, 
    use_amp=True, 
    epoch=None, 
    log_wandb=False, 
    global_step=None,
    perf_monitor=None
):
    """Memory-efficient training loop with performance monitoring."""
    model.train()
    running_loss = 0.0
    
    # Use numpy arrays instead of lists for efficiency
    all_targets = []
    all_probs = []
    
    optimizer.zero_grad(set_to_none=True)
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=use_amp)
    
    # Initialize performance monitor
    if perf_monitor is None:
        perf_monitor = PerformanceMonitor()
    perf_monitor.reset()
    
    # Progress bar
    desc = f"Epoch {epoch} [Train]" if epoch is not None else "Training"
    pbar = tqdm(enumerate(loader), total=len(loader), desc=desc, leave=False)
    
    if global_step is None:
        global_step = 0
    
    data_iter = iter(loader)
    
    for batch_idx in range(len(loader)):
        with perf_monitor.time_operation('batch'):
            # Time data loading
            with perf_monitor.time_operation('data_loading'):
                try:
                    imgs, labels, _ = next(data_iter)
                except StopIteration:
                    break
            
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass
            with perf_monitor.time_operation('forward'):
                with autocast(enabled=use_amp):
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                
                # Normalize loss for gradient accumulation
                loss = loss / accumulation_steps
            
            # Backward pass
            with perf_monitor.time_operation('backward'):
                scaler.scale(loss).backward()
            
            # Update weights
            with perf_monitor.time_operation('optimizer'):
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            
            # Accumulate loss
            running_loss += loss.item() * imgs.size(0) * accumulation_steps
            
            # Collect predictions - FIXED: Keep as numpy arrays
            with torch.no_grad():
                probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
                all_probs.append(probs)
                all_targets.append(labels.cpu().numpy())
            
            # CRITICAL: Explicitly delete tensors to free GPU memory
            del imgs, labels, logits, loss, probs
            
            # Update progress bar less frequently
            if batch_idx % 10 == 0:
                current_loss = running_loss / ((batch_idx + 1) * loader.batch_size)
                mem_stats = perf_monitor.get_memory_stats()
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'gpu_mb': f'{mem_stats.get("gpu_allocated_mb", 0):.0f}'
                })
            
            # Log batch metrics to wandb less frequently
            if log_wandb and batch_idx % 50 == 0:
                try:
                    mem_stats = perf_monitor.get_memory_stats()
                    wandb.log({
                        'batch/train_loss': running_loss / ((batch_idx + 1) * loader.batch_size),
                        'batch/step': global_step + batch_idx,
                        'batch/gpu_mb': mem_stats.get('gpu_allocated_mb', 0),
                        'batch/cpu_ram_mb': mem_stats.get('cpu_ram_mb', 0),
                    }, step=global_step + batch_idx)
                except Exception:
                    pass
        
        pbar.update(1)
    
    pbar.close()
    
    # Handle remaining gradients
    if len(loader) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    
    # FIXED: Concatenate numpy arrays efficiently
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    
    # Calculate metrics
    epoch_loss = running_loss / len(loader.dataset)
    try:
        epoch_auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        epoch_auc = float('nan')
    
    preds = (all_probs >= 0.5).astype(int)
    epoch_acc = accuracy_score(all_targets, preds)
    
    # Get performance stats
    perf_stats = perf_monitor.get_throughput_stats(len(loader.dataset))
    mem_stats = perf_monitor.get_memory_stats()
    
    # Clean up
    del all_probs, all_targets, preds
    gc.collect()
    
    return {
        'loss': epoch_loss, 
        'auc': epoch_auc, 
        'acc': epoch_acc, 
        'global_step': global_step + len(loader),
        'perf': perf_stats,
        'memory': mem_stats,
    }


# ============================================================================
# MEMORY-EFFICIENT EVALUATION
# ============================================================================

def evaluate(model, loader, criterion, device, use_amp=True, epoch=None, perf_monitor=None):
    """Memory-efficient evaluation with performance monitoring."""
    model.eval()
    running_loss = 0.0
    
    all_targets = []
    all_probs = []
    
    if perf_monitor is None:
        perf_monitor = PerformanceMonitor()
    perf_monitor.reset()
    
    desc = f"Epoch {epoch} [Val]" if epoch is not None else "Evaluating"
    pbar = tqdm(enumerate(loader), total=len(loader), desc=desc, leave=False)
    
    with torch.no_grad():
        for batch_idx, (imgs, labels, _) in pbar:
            with perf_monitor.time_operation('batch'):
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                with autocast(enabled=use_amp):
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                
                running_loss += loss.item() * imgs.size(0)
                
                probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
                all_probs.append(probs)
                all_targets.append(labels.cpu().numpy())
                
                # Clean up
                del imgs, labels, logits, loss, probs
            
            if batch_idx % 10 == 0:
                current_loss = running_loss / ((batch_idx + 1) * loader.batch_size)
                pbar.set_postfix({'loss': f'{current_loss:.4f}'})
            
            pbar.update(1)
    
    pbar.close()
    
    # Concatenate results
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    
    # Calculate metrics
    epoch_loss = running_loss / len(loader.dataset)
    try:
        epoch_auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        epoch_auc = float('nan')
    
    preds = (all_probs >= 0.5).astype(int)
    epoch_acc = accuracy_score(all_targets, preds)
    cm = confusion_matrix(all_targets, preds)
    cls_report = classification_report(
        all_targets, preds, 
        target_names=['HER2-', 'HER2+'], 
        output_dict=True,
        zero_division=0
    )
    
    # Get performance stats
    perf_stats = perf_monitor.get_throughput_stats(len(loader.dataset))
    mem_stats = perf_monitor.get_memory_stats()
    
    result = {
        'loss': epoch_loss,
        'auc': epoch_auc,
        'acc': epoch_acc,
        'cm': cm,
        'report': cls_report,
        'targets': all_targets.tolist(),
        'probs': all_probs.tolist(),
        'perf': perf_stats,
        'memory': mem_stats,
    }
    
    # Clean up
    del all_probs, all_targets, preds
    gc.collect()
    
    return result


# ============================================================================
# UTILITIES
# ============================================================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_metrics_csv_json(history, best_metrics, logs_dir: Path):
    """Save training metrics to CSV and JSON."""
    rows = []
    for e, rec in enumerate(history):
        row = {'epoch': e}
        row.update({f'train_{k}': rec['train'][k] for k in ['loss', 'auc', 'acc']})
        row.update({f'val_{k}': rec['val'][k] for k in ['loss', 'auc', 'acc']})
        
        # Add performance metrics if available
        if 'perf' in rec['train']:
            row['train_samples_per_sec'] = rec['train']['perf'].get('samples_per_sec', 0)
        if 'perf' in rec['val']:
            row['val_samples_per_sec'] = rec['val']['perf'].get('samples_per_sec', 0)
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(logs_dir / 'metrics.csv', index=False)
    
    with open(logs_dir / 'best.json', 'w') as f:
        json.dump(best_metrics, f, indent=2)


def _with_defaults(user_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Merge user config with defaults."""
    defaults = {
        'train_csv': 'outputs/patches_index_train.csv',
        'val_csv': 'outputs/patches_index_val.csv',
        'output_dir': 'outputs/phase1',
        'pretrained': True,
        'input_size': 512,
        'batch_size': 32,
        'num_workers': 4,
        'epochs': 10,
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'label_col': 'label',
        'path_col': 'path',
        'save_best_by': 'auc',
        'seed': 42,
        'accumulation_steps': 1,
        'use_amp': True,
        'resume': True,
        'use_wandb': True,
        'persistent_workers': True,  # NEW: Keep workers alive
        'prefetch_factor': 2,  # NEW: Prefetch batches
    }
    cfg = {**defaults, **(user_cfg or {})}
    return cfg


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_phase1(config: Dict[str, Any]):
    """
    Run Phase 1 training (ResNet-50) with memory leak fixes and performance monitoring.
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        dict with training results and paths
    """
    cfg = _with_defaults(config)
    
    set_seed(int(cfg.get('seed', 42)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directories
    out_dir = Path(cfg['output_dir'])
    models_dir = out_dir / 'models'
    logs_dir = out_dir / 'logs'
    vis_dir = out_dir / 'gradcam'
    tb_dir = out_dir / 'tensorboard'
    for d in [out_dir, models_dir, logs_dir, vis_dir, tb_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=str(tb_dir))
    
    # Performance monitor
    perf_monitor = PerformanceMonitor()
    
    # Define checkpoint path
    checkpoint_path = models_dir / 'checkpoint_last.pth'
    resume_training = cfg.get('resume', True)
    
    # Initialize Weights & Biases
    use_wandb = cfg.get('use_wandb', True) and WANDB_AVAILABLE
    wandb_id = None
    
    if use_wandb:
        wandb_project = cfg.get('wandb_project', 'her2-classification')
        wandb_name = cfg.get('wandb_name', f"phase1_bs{cfg['batch_size']}_size{cfg['input_size']}")
        
        if resume_training and checkpoint_path.exists():
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                wandb_id = checkpoint.get('wandb_id', None)
                if wandb_id:
                    print(f"Resuming wandb run: {wandb_id}")
            except Exception:
                pass
        
        try:
            wandb.init(
                project=wandb_project,
                name=wandb_name,
                config=cfg,
                dir=str(out_dir),
                tags=['phase1', 'resnet50', 'her2-classifier', 'optimized'],
                id=wandb_id,
                resume='allow' if wandb_id else None,
                settings=wandb.Settings(
                    code_dir=None,
                    _disable_stats=False,
                    _disable_meta=False,
                )
            )
            wandb_id = wandb.run.id
            print(f"Weights & Biases initialized: {wandb_project}/{wandb_name}")
        except Exception as e:
            print(f"Warning: Failed to initialize Weights & Biases: {e}")
            use_wandb = False
    
    # Build transforms
    train_tfms = build_transforms(int(cfg['input_size']), train=True)
    val_tfms = build_transforms(int(cfg['input_size']), train=False)
    
    # Build datasets
    train_set = PatchDataset(
        cfg['train_csv'], 
        cfg['path_col'], 
        cfg['label_col'], 
        transform=train_tfms
    )
    val_set = PatchDataset(
        cfg['val_csv'], 
        cfg['path_col'], 
        cfg['label_col'], 
        transform=val_tfms
    )
    
    # OPTIMIZED: DataLoader with better settings
    pin_mem = torch.cuda.is_available()
    persistent_workers = cfg.get('persistent_workers', True) and int(cfg['num_workers']) > 0
    
    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg['batch_size']),
        shuffle=True,
        num_workers=int(cfg['num_workers']),
        pin_memory=pin_mem,
        persistent_workers=persistent_workers,
        prefetch_factor=cfg.get('prefetch_factor', 2) if int(cfg['num_workers']) > 0 else None,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(cfg['batch_size']),
        shuffle=False,
        num_workers=int(cfg['num_workers']),
        pin_memory=pin_mem,
        persistent_workers=persistent_workers,
        prefetch_factor=cfg.get('prefetch_factor', 2) if int(cfg['num_workers']) > 0 else None,
    )
    
    # Build model
    model, last_conv = build_resnet50(cfg.get('pretrained', True))
    model = model.to(device)
    
    # OPTIMIZED: Compile model if PyTorch 2.0+
    if hasattr(torch, 'compile') and cfg.get('compile', False):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model, mode='default')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        model, 
        lr=float(cfg['lr']), 
        weight_decay=float(cfg['weight_decay'])
    )
    
    # FIXED: Use gradient logging only (not 'all' which leaks memory)
    if use_wandb:
        wandb.watch(model, criterion, log='gradients', log_freq=100)
        print(f"Model watching enabled - tracking gradients only")
    
    # Training state
    best_key = cfg.get('save_best_by', 'auc')
    if best_key not in ('auc', 'acc'):
        best_key = 'auc'
    
    best_score = -np.inf
    best_metrics = {}
    history = []
    start_epoch = 0
    
    # Resume from checkpoint
    if resume_training and checkpoint_path.exists():
        print(f"Found checkpoint at {checkpoint_path}, resuming training...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_score = checkpoint['best_score']
            best_metrics = checkpoint.get('best_metrics', {})
            # FIXED: Only load recent history (last 10 epochs) to prevent memory bloat
            full_history = checkpoint.get('history', [])
            history = full_history[-10:] if len(full_history) > 10 else full_history
            
            print(f"Resumed from epoch {checkpoint['epoch']}")
            print(f"Best {best_key} so far: {best_score:.4f}")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}")
            start_epoch = 0
    
    # Training settings
    accumulation_steps = int(cfg.get('accumulation_steps', 1))
    use_amp = cfg.get('use_amp', True) and torch.cuda.is_available()
    
    if accumulation_steps > 1:
        effective_bs = int(cfg['batch_size']) * accumulation_steps
        print(f"Using gradient accumulation: {accumulation_steps} steps (effective batch size: {effective_bs})")
    if use_amp:
        print("Using mixed precision training (FP16)")
    
    # Start training
    start_time = time.time()
    epochs = int(cfg['epochs'])
    global_step = 0
    
    epoch_pbar = tqdm(
        range(start_epoch, epochs), 
        desc="Training Progress", 
        initial=start_epoch, 
        total=epochs
    )

    # --- ADDED: Epoch loop, logging, checkpointing, final save & cleanup ---
    for epoch in epoch_pbar:
        t_epoch_start = time.time()

        # Train for one epoch (performance monitor passed through)
        train_res = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            accumulation_steps=accumulation_steps,
            use_amp=use_amp,
            epoch=epoch,
            log_wandb=use_wandb,
            global_step=global_step,
            perf_monitor=perf_monitor
        )

        global_step = train_res.get('global_step', global_step)

        # Evaluate
        val_res = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            epoch=epoch,
            perf_monitor=perf_monitor
        )

        # Compose epoch record
        rec = {
            'train': {
                'loss': float(train_res['loss']),
                'auc': float(train_res['auc']),
                'acc': float(train_res['acc']),
                'perf': train_res.get('perf', {}),
                'memory': train_res.get('memory', {})
            },
            'val': {
                'loss': float(val_res['loss']),
                'auc': float(val_res['auc']),
                'acc': float(val_res['acc']),
                'perf': val_res.get('perf', {}),
                'memory': val_res.get('memory', {})
            }
        }
        history.append(rec)

        # TensorBoard logging
        try:
            writer.add_scalar('train/loss', rec['train']['loss'], epoch)
            writer.add_scalar('train/auc', rec['train']['auc'], epoch)
            writer.add_scalar('train/acc', rec['train']['acc'], epoch)
            writer.add_scalar('val/loss', rec['val']['loss'], epoch)
            writer.add_scalar('val/auc', rec['val']['auc'], epoch)
            writer.add_scalar('val/acc', rec['val']['acc'], epoch)

            # Perf metrics
            writer.add_scalar('perf/train_samples_per_sec', rec['train']['perf'].get('samples_per_sec', 0), epoch)
            writer.add_scalar('perf/val_samples_per_sec', rec['val']['perf'].get('samples_per_sec', 0), epoch)
            writer.add_scalar('mem/gpu_allocated_mb', rec['val']['memory'].get('gpu_allocated_mb', 0), epoch)
            writer.flush()
        except Exception:
            pass

        # WandB logging (if available)
        if use_wandb:
            try:
                log_data = {
                    'epoch': epoch,
                    'train/loss': rec['train']['loss'],
                    'train/auc': rec['train']['auc'],
                    'train/acc': rec['train']['acc'],
                    'val/loss': rec['val']['loss'],
                    'val/auc': rec['val']['auc'],
                    'val/acc': rec['val']['acc'],
                    'perf/train_samples_per_sec': rec['train']['perf'].get('samples_per_sec', 0),
                    'perf/val_samples_per_sec': rec['val']['perf'].get('samples_per_sec', 0),
                    'mem/gpu_allocated_mb': rec['val']['memory'].get('gpu_allocated_mb', 0),
                    'mem/cpu_ram_mb': rec['val']['memory'].get('cpu_ram_mb', 0),
                    'global_step': global_step,
                }
                wandb.log(log_data, step=epoch)
            except Exception:
                pass

        # Checkpointing
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_score': best_score,
                'best_metrics': best_metrics,
                'history': history,
                'wandb_id': wandb_id,
            }
            torch.save(checkpoint, checkpoint_path)
        except Exception as e:
            print(f"Warning: Failed to save checkpoint: {e}")

        # Save best model if improved
        val_key_val = val_res.get(best_key, float('-inf'))
        if val_key_val is not None and val_key_val > best_score:
            best_score = float(val_key_val)
            best_metrics = {
                'epoch': epoch,
                'val_loss': float(val_res['loss']),
                'val_auc': float(val_res['auc']),
                'val_acc': float(val_res['acc']),
                'perf': val_res.get('perf', {}),
                'memory': val_res.get('memory', {})
            }
            try:
                best_path = models_dir / 'model_best.pth'
                torch.save({'model_state_dict': model.state_dict(), 'best_metrics': best_metrics}, best_path)
            except Exception as e:
                print(f"Warning: Failed to save best model: {e}")

        # Epoch summary for progress bar
        epoch_time = time.time() - t_epoch_start
        epoch_pbar.set_postfix({
            'train_loss': f"{rec['train']['loss']:.4f}",
            'val_loss': f"{rec['val']['loss']:.4f}",
            'val_auc': f"{rec['val']['auc']:.4f}",
            'epoch_s': f"{epoch_time:.1f}"
        })
        epoch_pbar.update(0)

        # Cleanup to avoid memory growth
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # After training: save final metrics and cleanup
    try:
        save_metrics_csv_json(history, best_metrics, logs_dir)
    except Exception as e:
        print(f"Warning: Failed to save metrics files: {e}")

    try:
        writer.close()
    except Exception:
        pass

    if use_wandb:
        try:
            wandb.run.summary.update(best_metrics or {})
            wandb.finish()
        except Exception:
            pass

    # Final return for callers
    return {
        'output_dir': str(out_dir),
        'models_dir': str(models_dir),
        'logs_dir': str(logs_dir),
        'history': history,
        'best_metrics': best_metrics,
    }