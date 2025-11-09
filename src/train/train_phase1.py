"""
Phase 1 patch-level training entrypoint (ResNet-50) with DDP support.

Usage:
    # Single GPU/CPU
    from src.train.train_phase1 import train_phase1
    results = train_phase1(config)
    
    # Multi-GPU DDP (recommended)
    torchrun --nproc_per_node=2 -m src.train.train_phase1 --config configs/config.yaml

Expected config keys (with defaults):
    - train_csv: str, path to training CSV with columns [path, label]
    - val_csv: str, path to validation CSV
    - output_dir: str, directory to save models/logs/visualizations
    - pretrained: bool, load ImageNet weights for ResNet-50
    - input_size: int, resize side for patches (e.g., 224 or 512)
    - batch_size: int (per GPU)
    - num_workers: int
    - epochs: int
    - lr: float
    - weight_decay: float
    - label_col: str, name of label column (default 'label')
    - path_col: str, name of image path column (default 'path')
    - save_best_by: 'auc' or 'acc'
    - seed: int, random seed (default 42)
    - use_amp: bool, enable mixed precision (default True on CUDA)
    - accumulation_steps: int, gradient accumulation (default 1)
    - use_ddp: bool, enable distributed training (default False)
    - use_focal_loss: bool, use focal loss for imbalanced data (default False)
    - early_stopping_patience: int, epochs to wait before stopping (default 0=disabled)
"""

from __future__ import annotations

import os
import json
import time
import random
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
import torchvision.transforms as T
import torchvision.models as models
import gc
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report

# Stain normalization (Macenko)
from src.preprocessing.stain_normalization import macenko_normalization

# Weights & Biases for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Distributed utilities
def setup_distributed():
    """Initialize distributed training if launched via torchrun.
    Returns (rank, world_size, local_rank). Falls back to (0,1,0) if not distributed.
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            backend = 'nccl'
        else:
            backend = 'gloo'
        torch.distributed.init_process_group(backend=backend, init_method='env://')
        return rank, world_size, local_rank
    return 0, 1, 0

def cleanup_distributed():
    """Clean up distributed process group."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

def is_main_process(rank: int = 0) -> bool:
    """Check if current process is main (rank 0)."""
    return rank == 0


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MacenkoNormalizeTransform:
    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        try:
            norm = macenko_normalization(arr)
        except Exception:
            norm = arr
        return Image.fromarray(norm)


def build_transforms(input_size: int, train: bool = True, enable_stain_norm: bool = True):
    tfms = []
    if enable_stain_norm:
        tfms.append(MacenkoNormalizeTransform())
    tfms.append(T.Resize((input_size, input_size)))
    if train:
        tfms.extend([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        ])
    tfms.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return T.Compose(tfms)


class PatchDataset(Dataset):
    def __init__(self, csv_path: str, path_col: str, label_col: str, transform=None):
        self.df = pd.read_csv(csv_path)
        self.path_col = path_col
        self.label_col = label_col if label_col in self.df.columns else 'target'
        self.transform = transform
        if self.path_col not in self.df.columns:
            raise ValueError(f"Missing column '{self.path_col}' in {csv_path}")
        if self.label_col not in self.df.columns:
            raise ValueError(f"Missing label column '{label_col}' or 'target' in {csv_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row[self.path_col]
        label = int(row[self.label_col])
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label, path
def build_resnet50(pretrained: bool = True, num_classes: int = 2):
    """Build ResNet-50 with custom classification head."""
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
    except Exception:
        model = models.resnet50(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_optimizer(model, lr=1e-4, weight_decay=1e-4):
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance (medical applications).
    
    Focuses on hard-to-classify examples, improves recall on minority class.
    Recommended for HER2 when positive cases are <30% of dataset.
    
    Args:
        alpha: Weighting factor in [0,1] for class 1 (positive). Default 0.25.
        gamma: Focusing parameter >= 0. gamma=0 is equivalent to CE. Default 2.0.
        class_weights: Optional per-class weights tensor of shape [num_classes].
    """
    def __init__(self, alpha=0.25, gamma=2.0, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.class_weights)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def build_criterion(loss_type: str = 'cross_entropy', class_weights=None):
    """Build loss function with proper class weighting for medical data.
    
    Args:
        loss_type: 'cross_entropy' or 'focal'
        class_weights: optional tensor/list of shape [num_classes]
    
    Returns:
        Loss criterion (nn.Module)
    
    Note: Focal loss can also accept class_weights for combined effect.
    """
    if class_weights is not None and not torch.is_tensor(class_weights):
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    if loss_type == 'focal':
        # Focal loss with optional class weighting
        return FocalLoss(alpha=0.25, gamma=2.0, class_weights=class_weights)
    
    # Standard cross-entropy with optional class weighting
    if class_weights is not None:
        return nn.CrossEntropyLoss(weight=class_weights)
    return nn.CrossEntropyLoss()


def broadcast_model_parameters(model):
    """Broadcast parameters and buffers from rank 0 to all ranks."""
    if not torch.distributed.is_initialized():
        return
    module = model.module if isinstance(model, DDP) else model
    for tensor in list(module.parameters()) + list(module.buffers()):
        torch.distributed.broadcast(tensor.data, src=0)


def aggregate_metrics_ddp(local_loss, local_count, all_targets, all_probs, device):
    """Aggregate metrics across DDP processes."""
    if torch.distributed.is_initialized():
        loss_sum = torch.tensor(local_loss, device=device)
        count_sum = torch.tensor(local_count, device=device, dtype=torch.long)
        torch.distributed.all_reduce(loss_sum, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(count_sum, op=torch.distributed.ReduceOp.SUM)
        epoch_loss = (loss_sum / count_sum.clamp(min=1)).item()
        
        gathered_targets = [None] * torch.distributed.get_world_size()
        gathered_probs = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered_targets, all_targets)
        torch.distributed.all_gather_object(gathered_probs, all_probs)
        all_targets = [x for sub in gathered_targets for x in sub]
        all_probs = [x for sub in gathered_probs for x in sub]
    else:
        epoch_loss = local_loss / max(local_count, 1)
    
    return epoch_loss, all_targets, all_probs


def train_one_epoch(model, loader, criterion, optimizer, device, rank=0, accumulation_steps=1, use_amp=True, epoch=None, log_wandb=False, global_step=None, grad_clip_norm: float = 0.0, scaler: GradScaler | None = None):
    """Optimized training loop with DDP support."""
    model.train()
    running_loss = 0.0
    sample_count = 0
    all_targets, all_probs = [], []
    optimizer.zero_grad(set_to_none=True)
    
    scaler = scaler or GradScaler(enabled=use_amp)
    desc = f"Epoch {epoch} [Train]" if epoch is not None else "Training"
    pbar = tqdm(enumerate(loader), total=len(loader), desc=desc, leave=False, disable=not is_main_process(rank))
    
    if global_step is None:
        global_step = 0
    
    for batch_idx, (imgs, labels, _) in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Forward with autocast (handles AMP enabled/disabled internally)
        with autocast(enabled=use_amp):
            logits = model(imgs)
            loss = criterion(logits, labels) / accumulation_steps
        
        # Backward
        scaler.scale(loss).backward()
        
        # Optimizer step with gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            if grad_clip_norm and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        # Metrics tracking
        batch_size = imgs.size(0)
        running_loss += loss.item() * batch_size * accumulation_steps
        sample_count += batch_size
        with torch.no_grad():
            probs = torch.softmax(logits.float(), dim=1)[:, 1]
        all_probs.extend(probs.cpu().tolist())
        all_targets.extend(labels.cpu().tolist())
        
        # Progress bar update (calculate average loss correctly)
        if is_main_process(rank):
            avg_loss = running_loss / max(sample_count, 1)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # Periodic wandb logging (main process only)
        if log_wandb and is_main_process(rank) and batch_idx % 10 == 0:
            avg_loss = running_loss / max(sample_count, 1)
            wandb.log({'batch/train_loss': avg_loss, 'batch/step': global_step + batch_idx}, step=global_step + batch_idx)
    
    pbar.close()
    
    # Handle final accumulation step
    if len(loader) % accumulation_steps != 0:
        if grad_clip_norm and grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    
    # Aggregate metrics
    epoch_loss, global_targets, global_probs = aggregate_metrics_ddp(
        running_loss, sample_count, all_targets, all_probs, device
    )
    
    try:
        epoch_auc = roc_auc_score(global_targets, global_probs)
        preds = (np.array(global_probs) >= 0.5).astype(int)
        epoch_acc = accuracy_score(global_targets, preds)
    except Exception:
        epoch_auc = float('nan')
        epoch_acc = float('nan')
    
    return {'loss': epoch_loss, 'auc': epoch_auc, 'acc': epoch_acc, 'global_step': global_step + len(loader)}


def evaluate(model, loader, criterion, device, rank=0, use_amp=True, epoch=None):
    """Optimized evaluation with DDP support and global AUC on rank 0.
    Uses all_gather_object to aggregate predictions/targets across processes.
    """
    model.eval()
    running_loss = 0.0
    sample_count = 0
    all_targets, all_probs = [], []
    
    desc = f"Epoch {epoch} [Val]" if epoch is not None else "Evaluating"
    pbar = tqdm(enumerate(loader), total=len(loader), desc=desc, leave=False, disable=not is_main_process(rank))
    
    with torch.no_grad():
        for batch_idx, (imgs, labels, _) in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast(enabled=use_amp):
                logits = model(imgs)
                loss = criterion(logits, labels)
            batch_size = imgs.size(0)
            running_loss += loss.item() * batch_size
            sample_count += batch_size
            probs = torch.softmax(logits.float(), dim=1)[:, 1]
            all_probs.extend(probs.cpu().tolist())
            all_targets.extend(labels.cpu().tolist())
            if is_main_process(rank):
                avg_loss = running_loss / max(sample_count, 1)
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    pbar.close()
    
    # Aggregate metrics
    epoch_loss, global_targets, global_probs = aggregate_metrics_ddp(
        running_loss, sample_count, all_targets, all_probs, device
    )
    
    # Compute metrics (on rank 0 for reporting)
    # Compute optimal threshold (Youden's J)
    def _optimal_threshold(y_true, y_prob):
        """Compute optimal threshold using Youden's J statistic.
        
        Maximizes (sensitivity + specificity - 1) for balanced classification.
        Critical for medical applications where both false positives and 
        false negatives have clinical consequences.
        """
        try:
            from sklearn.metrics import roc_curve
            fpr, tpr, thr = roc_curve(y_true, y_prob)
            j = tpr - fpr  # Youden's J = sensitivity + specificity - 1
            idx = np.argmax(j)
            return float(thr[idx])
        except Exception as e:
            # Fallback to 0.5 if computation fails (shouldn't happen with valid data)
            return 0.5

    opt_thr = _optimal_threshold(global_targets, global_probs)
    # Compute metrics with proper error handling
    try:
        epoch_auc = roc_auc_score(global_targets, global_probs)
    except Exception:
        epoch_auc = float('nan')
    
    # Predictions at standard 0.5 threshold
    preds = (np.array(global_probs) >= 0.5).astype(int)
    # Predictions at optimal threshold (Youden's J)
    preds_opt = (np.array(global_probs) >= opt_thr).astype(int)
    
    # Compute metrics for both thresholds
    valid_data = len(global_targets) == len(preds) and len(preds) > 0
    epoch_acc = accuracy_score(global_targets, preds) if valid_data else float('nan')
    acc_opt = accuracy_score(global_targets, preds_opt) if valid_data else float('nan')
    
    # Confusion matrices (for sensitivity/specificity calculation)
    cm = confusion_matrix(global_targets, preds, labels=[0, 1]) if np.isfinite(epoch_acc) else np.array([[0, 0], [0, 0]])
    cm_opt = confusion_matrix(global_targets, preds_opt, labels=[0, 1]) if np.isfinite(acc_opt) else np.array([[0, 0], [0, 0]])
    
    # Detailed classification reports
    cls_report = classification_report(global_targets, preds, target_names=['HER2-', 'HER2+'], output_dict=True, zero_division=0) if np.isfinite(epoch_acc) else {}
    cls_report_opt = classification_report(global_targets, preds_opt, target_names=['HER2-', 'HER2+'], output_dict=True, zero_division=0) if np.isfinite(acc_opt) else {}
    
    # Medical-grade metrics: Sensitivity and Specificity (standard and optimal threshold)
    sensitivity = float('nan')
    specificity = float('nan')
    sensitivity_opt = float('nan')
    specificity_opt = float('nan')
    if cm.sum() > 0 and np.isfinite(epoch_acc):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    if cm_opt.sum() > 0 and np.isfinite(acc_opt):
        tn_o, fp_o, fn_o, tp_o = cm_opt.ravel()
        sensitivity_opt = tp_o / (tp_o + fn_o) if (tp_o + fn_o) > 0 else 0.0
        specificity_opt = tn_o / (tn_o + fp_o) if (tn_o + fp_o) > 0 else 0.0
    
    return {
        'loss': epoch_loss,
        'auc': epoch_auc,
        'acc': epoch_acc,
        'sensitivity': sensitivity,  # True positive rate (recall for HER2+)
        'specificity': specificity,  # True negative rate
        'sensitivity_opt': sensitivity_opt,
        'specificity_opt': specificity_opt,
        'cm': cm,
        'cm_opt': cm_opt,
        'report': cls_report,
        'report_opt': cls_report_opt,
        'opt_threshold': opt_thr,
        'targets': global_targets,
        'probs': global_probs,
    }


def save_metrics_csv_json(history, best_metrics, logs_dir: Path):
    """Save training history and best metrics to disk."""
    rows = [
        {'epoch': e, **{f'train_{k}': rec['train'][k] for k in ['loss', 'auc', 'acc']},
         **{f'val_{k}': rec['val'][k] for k in ['loss', 'auc', 'acc']}}
        for e, rec in enumerate(history)
    ]
    pd.DataFrame(rows).to_csv(logs_dir / 'metrics.csv', index=False)
    (logs_dir / 'best.json').write_text(json.dumps(best_metrics, indent=2))


def save_checkpoint(epoch, model, optimizer, scheduler, best_score, best_metrics, history, 
                    patience_counter, cfg, wandb_id, checkpoint_path, use_ddp):
    """Save training checkpoint."""
    model_state = model.module.state_dict() if use_ddp else model.state_dict()
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_score': best_score,
        'best_metrics': best_metrics,
        'history': history,
        'patience_counter': patience_counter,
        'config': cfg,
        'wandb_id': wandb_id,
    }, checkpoint_path)


def log_wandb_epoch(train_log, val_log, epoch, optimizer, use_wandb):
    """Log epoch metrics to Weights & Biases."""
    if not use_wandb:
        return
    
    wandb_log = {
        'epoch': epoch + 1,
        'train/loss': train_log['loss'],
        'train/auc': train_log['auc'],
        'train/accuracy': train_log['acc'],
        'val/loss': val_log['loss'],
        'val/auc': val_log['auc'],
        'val/accuracy': val_log['acc'],
        'learning_rate': optimizer.param_groups[0]['lr'],
    }
    
    # Medical metrics
    for metric in ['sensitivity', 'specificity', 'opt_threshold', 'sensitivity_opt', 'specificity_opt']:
        if metric in val_log and np.isfinite(val_log[metric]):
            wandb_log[f'val/{metric}'] = val_log[metric]
    
    # GPU metrics
    if torch.cuda.is_available():
        wandb_log.update({
            'gpu/memory_allocated_gb': torch.cuda.memory_allocated() / 1024**3,
            'gpu/memory_reserved_gb': torch.cuda.memory_reserved() / 1024**3,
            'gpu/max_memory_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3,
        })
    
    wandb.log(wandb_log)


def update_wandb_best_summary(val_log, epoch):
    """Update W&B run summary with best metrics."""
    wandb.run.summary.update({
        'best_epoch': epoch,
        'best_val_auc': float(val_log['auc']),
        'best_val_acc': float(val_log['acc']),
        'best_val_loss': float(val_log['loss']),
        'best_val_opt_threshold': float(val_log.get('opt_threshold', 0.5)),
    })
    
    # Medical metrics
    for metric in ['sensitivity', 'specificity', 'sensitivity_opt', 'specificity_opt']:
        if metric in val_log and np.isfinite(val_log[metric]):
            wandb.run.summary[f'best_val_{metric}'] = float(val_log[metric])
    
    # Per-class metrics from optimal threshold report
    if 'report_opt' in val_log and isinstance(val_log['report_opt'], dict):
        her2_pos = val_log['report_opt'].get('HER2+', {})
        for k in ['precision', 'recall', 'f1-score']:
            if k in her2_pos:
                wandb.run.summary[f'best_val_pos_{k.replace("-", "_")}'] = her2_pos[k]


def _with_defaults(user_cfg: Dict[str, Any]) -> Dict[str, Any]:
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
        'save_best_by': 'auc',  # or 'acc'
        'seed': 42,
    }
    cfg = {**defaults, **(user_cfg or {})}
    return cfg


def calculate_class_weights(train_csv, label_col):
    """Robust class weight calculation for binary labels.

    Ensures both classes 0 and 1 have non-zero counts (floors to 1) to avoid
    division by zero in extreme imbalance scenarios.
    """
    df = pd.read_csv(train_csv)
    labels = df[label_col].astype(int)
    class_counts = np.array([
        max((labels == 0).sum(), 1),
        max((labels == 1).sum(), 1)
    ], dtype=np.float32)
    total = class_counts.sum()
    weights = total / (2 * class_counts)
    return torch.tensor(weights, dtype=torch.float32)


def train_phase1(config: Dict[str, Any]):
    """Run Phase 1 training (ResNet-50) with DDP support and medical-grade optimizations.

    Args:
        config: dict containing training configuration. See module docstring for keys.

    Returns:
        dict with keys: best_model_path, best_metrics, logs_dir, models_dir
    """
    cfg = _with_defaults(config)
    
    # Initialize DDP if enabled
    use_ddp = cfg.get('use_ddp', False)
    rank = 0
    world_size = 1
    local_rank = 0
    if use_ddp:
        rank, world_size, local_rank = setup_distributed()
    
    # Extract commonly used config values once
    input_size = int(cfg['input_size'])
    batch_size = int(cfg['batch_size'])
    num_workers = int(cfg['num_workers'])
    epochs = int(cfg['epochs'])
    lr = float(cfg['lr'])
    weight_decay = float(cfg['weight_decay'])
    seed = int(cfg.get('seed', 42))
    
    set_seed(seed)
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')

    out_dir = Path(cfg['output_dir'])
    models_dir = out_dir / 'models'
    logs_dir = out_dir / 'logs'
    tb_dir = out_dir / 'tensorboard'
    if is_main_process(rank):
        for d in [out_dir, models_dir, logs_dir, tb_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    # Initialize TensorBoard and W&B on main process only
    writer = SummaryWriter(log_dir=str(tb_dir)) if is_main_process(rank) else None
    checkpoint_path = models_dir / 'checkpoint_last.pth'
    resume_training = cfg.get('resume', True)
    
    use_wandb = cfg.get('use_wandb', True) and WANDB_AVAILABLE and is_main_process(rank)
    wandb_id = None
    if use_wandb:
        wandb_project = cfg.get('wandb_project', 'her2-classification')
        wandb_name = cfg.get('wandb_name', f"phase1_bs{batch_size}_size{input_size}")
        
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
                tags=['phase1', 'resnet50', 'her2-classifier'],
                id=wandb_id,
                resume='allow' if wandb_id else None,
                settings=wandb.Settings(code_dir=None, _disable_stats=False, _disable_meta=False)
            )
            wandb_id = wandb.run.id
            print(f"Weights & Biases initialized: {wandb_project}/{wandb_name}")
        except Exception as e:
            print(f"Warning: Failed to initialize Weights & Biases: {e}")
            print("Continuing with TensorBoard logging only")
            use_wandb = False
    elif is_main_process(rank):
        if not WANDB_AVAILABLE:
            print("Weights & Biases not available - install with: pip install wandb")
        else:
            print("Weights & Biases logging disabled")
    
    # Datasets and loaders
    enable_stain_norm = bool(cfg.get('enable_stain_norm', True))
    train_tfms = build_transforms(input_size, train=True, enable_stain_norm=enable_stain_norm)
    val_tfms = build_transforms(input_size, train=False, enable_stain_norm=enable_stain_norm)

    train_set = PatchDataset(cfg['train_csv'], cfg['path_col'], cfg['label_col'], transform=train_tfms)
    val_set = PatchDataset(cfg['val_csv'], cfg['path_col'], cfg['label_col'], transform=val_tfms)

    # Use DistributedSampler for DDP
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp else None
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False) if use_ddp else None
    
    pin_mem = torch.cuda.is_available()
    prefetch_factor = int(cfg.get('prefetch_factor', 2))
    persistent_workers = bool(cfg.get('persistent_workers', True)) and num_workers > 0
    def _worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    # Build DataLoader kwargs
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': pin_mem,
        **(
            {
                'prefetch_factor': prefetch_factor,
                'persistent_workers': persistent_workers,
                'worker_init_fn': _worker_init_fn,
            }
            if num_workers > 0
            else {}
        ),
    }
    train_loader = DataLoader(train_set, shuffle=(train_sampler is None), sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, sampler=val_sampler, **loader_kwargs)

    # Model initialization
    model = build_resnet50(cfg.get('pretrained', True)).to(device)
    
    # Optional torch.compile for speed (PyTorch 2.x)
    if cfg.get('use_compile', False) and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode=cfg.get('compile_mode', 'default'))
            if is_main_process(rank):
                print(f"Model compiled (mode={cfg.get('compile_mode', 'default')})")
        except Exception as e:
            if is_main_process(rank):
                print(f"torch.compile failed: {e}")
    
    # Wrap in DDP if distributed
    if use_ddp:
        ddp_kwargs = {'device_ids': [local_rank], 'output_device': local_rank} if torch.cuda.is_available() else {}
        model = DDP(model, **ddp_kwargs)
        if is_main_process(rank):
            print(f"DDP enabled (world_size={world_size}, local_rank={local_rank})")
    
    # Medical-grade loss function with optional focal loss and class weighting
    loss_type = cfg.get('loss_type', 'cross_entropy')  # 'cross_entropy' or 'focal'
    use_class_weights = cfg.get('use_class_weights', True)
    class_weights = None
    if use_class_weights:
        class_weights = calculate_class_weights(cfg['train_csv'], cfg['label_col']).to(device)
        if is_main_process(rank):
            print(f"Using class weights for imbalanced dataset: {class_weights.tolist()}")
    
    criterion = build_criterion(loss_type=loss_type, class_weights=class_weights).to(device)
    optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay)
    
    # Cosine annealing LR scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    if use_wandb:
        wandb.watch(model, criterion, log='all', log_freq=100)
        if is_main_process(rank):
            print(f"Model watching enabled - tracking gradients and parameters")

    best_key = cfg.get('save_best_by', 'auc')
    if best_key not in ('auc', 'acc'):
        best_key = 'auc'
    best_score = -np.inf
    best_metrics = {}
    history = []
    start_epoch = 0
    patience_counter = 0
    early_stop_patience = cfg.get('early_stop_patience', cfg.get('early_stopping_patience', 10))  # Early stopping
    
    # Load checkpoint if resuming
    checkpoint_loaded = False
    if resume_training and checkpoint_path.exists() and is_main_process(rank):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            (model.module if use_ddp else model).load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint.get('scheduler_state_dict', scheduler.state_dict()))
            start_epoch = checkpoint['epoch'] + 1
            best_score = checkpoint['best_score']
            best_metrics = checkpoint.get('best_metrics', {})
            history = checkpoint.get('history', [])
            patience_counter = checkpoint.get('patience_counter', 0)
            print(f"Resumed from epoch {checkpoint['epoch']}, best {best_key}: {best_score:.4f}")
            checkpoint_loaded = True
            if use_ddp:
                torch.distributed.barrier()
                broadcast_model_parameters(model)
        except Exception as e:
            print(f"Checkpoint load failed: {e}. Starting fresh.")
    elif is_main_process(rank):
        print("Starting training from scratch" + (" (no checkpoint)" if resume_training else " (resume disabled)"))
    
    # Sync all DDP ranks
    if use_ddp:
        torch.distributed.barrier()
        start_epoch_tensor = torch.tensor(start_epoch, device=device)
        torch.distributed.broadcast(start_epoch_tensor, src=0)
        start_epoch = start_epoch_tensor.item()
        if not checkpoint_loaded:
            broadcast_model_parameters(model)
    
    accumulation_steps = int(cfg.get('accumulation_steps', 1))
    use_amp = cfg.get('use_amp', True) and torch.cuda.is_available()
    
    if is_main_process(rank):
        if accumulation_steps > 1:
            print(f"Using gradient accumulation with {accumulation_steps} steps (effective batch size: {batch_size * accumulation_steps})")
        if use_amp:
            print("Using mixed precision training (FP16) for memory efficiency")

    start_time = time.time()
    global_step = 0
    # Persistent GradScaler across epochs for numerical stability
    scaler = GradScaler(enabled=use_amp)
    
    # Set sampler epoch for DDP (ensures different shuffle each epoch)
    if use_ddp and train_sampler is not None:
        train_sampler.set_epoch(start_epoch)
    
    epoch_pbar = tqdm(range(start_epoch, epochs), desc="Training Progress", initial=start_epoch, total=epochs, disable=not is_main_process(rank))
    
    for epoch in epoch_pbar:
        # Set epoch for distributed sampler
        if use_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        t0 = time.time()
        train_log = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            rank,
            accumulation_steps,
            use_amp,
            epoch=epoch+1,
            log_wandb=use_wandb,
            global_step=global_step,
            grad_clip_norm=float(cfg.get('grad_clip_norm', 0.0)),
            scaler=scaler,
        )
        global_step = train_log.get('global_step', global_step)
        val_log = evaluate(model, val_loader, criterion, device, rank, use_amp, epoch=epoch+1)
        history.append({'train': {k: v for k, v in train_log.items() if k != 'global_step'}, 'val': {k: val_log[k] for k in ['loss', 'auc', 'acc']}})

        # Step LR scheduler
        scheduler.step()
        
        # Logging (main process only)
        if is_main_process(rank):
            for metric_name, metric_dict in [('Loss', 'loss'), ('AUC', 'auc'), ('Accuracy', 'acc')]:
                writer.add_scalar(f'{metric_name}/train', train_log[metric_dict], epoch)
                writer.add_scalar(f'{metric_name}/val', val_log[metric_dict], epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            log_wandb_epoch(train_log, val_log, epoch, optimizer, use_wandb)

        # Check improvement and early stopping
        cur_score = val_log.get(best_key, float('-inf'))
        improved = np.isfinite(cur_score) and cur_score > best_score
        if improved:
            best_score = cur_score
            patience_counter = 0
            # Save comprehensive best metrics for medical reporting
            best_metrics = {
                'epoch': epoch,
                'score_key': best_key,
                'score': float(best_score),
                'val': {
                    'loss': float(val_log['loss']),
                    'auc': float(val_log['auc']),
                    'acc': float(val_log['acc']),
                    'sensitivity': float(val_log.get('sensitivity', float('nan'))),
                    'specificity': float(val_log.get('specificity', float('nan'))),
                    'opt_threshold': float(val_log.get('opt_threshold', 0.5)),
                }
            }
            # Save best model (main process only)
            if is_main_process(rank):
                model_state = model.module.state_dict() if use_ddp else model.state_dict()
                torch.save(model_state, models_dir / 'model_phase1.pth')
                if use_wandb:
                    update_wandb_best_summary(val_log, epoch)
        else:
            patience_counter += 1
        
        # Progress update (main process only)
        if is_main_process(rank):
            epoch_pbar.set_postfix(train_auc=f'{train_log["auc"]:.4f}', val_auc=f'{val_log["auc"]:.4f}', 
                                  best='*' if improved else '')
            tqdm.write(
                f"Epoch {epoch+1}/{epochs} | train: L={train_log['loss']:.4f} AUC={train_log['auc']:.4f} "
                f"Acc={train_log['acc']:.4f} | val: L={val_log['loss']:.4f} AUC={val_log['auc']:.4f} "
                f"Acc={val_log['acc']:.4f} | LR={optimizer.param_groups[0]['lr']:.2e} {'*' if improved else ''} "
                f"({time.time()-t0:.1f}s)"
            )
        
        # Save checkpoint (main process only)
        if is_main_process(rank):
            save_checkpoint(epoch, model, optimizer, scheduler, best_score, best_metrics,
                          history, patience_counter, cfg, wandb_id if use_wandb else None,
                          models_dir / 'checkpoint_last.pth', use_ddp)
            if (epoch + 1) % 5 == 0:
                save_checkpoint(epoch, model, optimizer, scheduler, best_score, best_metrics,
                              history, patience_counter, cfg, wandb_id if use_wandb else None,
                              models_dir / f'checkpoint_epoch_{epoch}.pth', use_ddp)
                tqdm.write(f"Saved checkpoint at epoch {epoch}")
        
        # Early stopping check
        if patience_counter >= early_stop_patience:
            if is_main_process(rank):
                tqdm.write(f"\nEarly stopping triggered after {patience_counter} epochs without improvement")
            break
        
        # Memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    epoch_pbar.close()
    
    if is_main_process(rank) and writer is not None:
        writer.close()

    # Save logs and final metrics (main process only)
    if is_main_process(rank):
        save_metrics_csv_json(history, best_metrics, logs_dir)
        try:
            cm = val_log['cm']
            rep = val_log['report']
            pd.DataFrame(cm, index=['HER2-', 'HER2+'], columns=['Pred -', 'Pred +']).to_csv(logs_dir / 'confusion_matrix.csv')
            pd.DataFrame(rep).to_csv(logs_dir / 'classification_report.csv')
            
            if use_wandb:
                wandb.log({
                    "conf_mat": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=val_log['targets'],
                        preds=(np.array(val_log['probs']) >= 0.5).astype(int),
                        class_names=['HER2-', 'HER2+']
                    )
                })
                report_df = pd.DataFrame(rep).transpose()
                wandb.log({"classification_report": wandb.Table(dataframe=report_df)})
        except Exception as e:
            print(f"Warning: Failed to log final metrics: {e}")
    
    # Close wandb (main process only)
    if use_wandb and is_main_process(rank):
        wandb.finish()
        print("Weights & Biases run finished")
    
    # Cleanup DDP
    if use_ddp:
        cleanup_distributed()
        if is_main_process(rank):
            print("Distributed training cleanup complete")

    # Print summary (main process only)
    if is_main_process(rank):
        best_model_path = models_dir / 'model_phase1.pth'
        print('Best model saved at:', best_model_path)
        print('Training finished in', f"{(time.time()-start_time)/60:.1f} min")
        print(f'TensorBoard logs saved to: {tb_dir}')
        print(f'To view: tensorboard --logdir={tb_dir}')
        if use_wandb:
            print(f'Weights & Biases dashboard: {wandb.run.url}')
        print(f'\nBest {best_key}: {best_score:.4f} at epoch {best_metrics.get("epoch", -1)}')

    return {
        'best_model_path': str(models_dir / 'model_phase1.pth'),
        'best_metrics': best_metrics,
        'logs_dir': str(logs_dir),
        'models_dir': str(models_dir),
        'tb_dir': str(tb_dir),
        'wandb_url': wandb.run.url if use_wandb and is_main_process(rank) else None,
    }


if __name__ == "__main__":
    import argparse, yaml
    parser = argparse.ArgumentParser(description="Phase 1 HER2 patch-level training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--loss-type", type=str, default=None, help="Override loss_type (focal|cross_entropy)")
    parser.add_argument("--use-class-weights", type=str, default=None, help="Override use_class_weights (true|false)")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        user_cfg = yaml.safe_load(f)
    if args.loss_type:
        user_cfg['loss_type'] = args.loss_type
    if args.use_class_weights:
        user_cfg['use_class_weights'] = args.use_class_weights.lower() in ("true","1","yes")
    out = train_phase1(user_cfg)
    print("Training complete. Best model:", out['best_model_path'])
