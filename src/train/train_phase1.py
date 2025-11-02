"""
Phase 1 patch-level training entrypoint (ResNet-50).

Usage:
    from src.train.train_phase1 import train_phase1
    results = train_phase1(config)

Expected config keys (with defaults):
    - train_csv: str, path to training CSV with columns [path, label]
    - val_csv: str, path to validation CSV
    - output_dir: str, directory to save models/logs/visualizations
    - pretrained: bool, load ImageNet weights for ResNet-50
    - input_size: int, resize side for patches (e.g., 224 or 512)
    - batch_size: int
    - num_workers: int
    - epochs: int
    - lr: float
    - weight_decay: float
    - label_col: str, name of label column (default 'label')
    - path_col: str, name of image path column (default 'path')
    - save_best_by: 'auc' or 'acc'
    - seed: int, random seed (default 42)
Notice
	- Do not run this file as multithreaded job. If wish to do so, Please ensure completeness of loops first.
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
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
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
    print("Warning: wandb not installed. Install with: pip install wandb")


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


def build_transforms(input_size: int, train: bool = True):
    tfms = []
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
        self.label_col = label_col
        self.transform = transform
        if self.path_col not in self.df.columns:
            raise ValueError(f"Missing column '{self.path_col}' in {csv_path}")
        if self.label_col not in self.df.columns:
            if 'target' in self.df.columns:
                self.label_col = 'target'
            else:
                raise ValueError(f"Missing label column in {csv_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row[self.path_col]
        label = int(row[self.label_col])
        img = Image.open(path).convert('RGB')
        # Apply transforms if needed
        if self.transform:
            try:
                img_transformed = self.transform(img) # Might cause error here in case of if image was closed before tensor transformation concludes.
                return img_transformed, label, path
            finally:
                try:
                    img.close()
                except (AttributeError, RuntimeError):
                    # Image already closed or doesn't support close
                    pass
        return img, label, path
def build_resnet50(pretrained: bool = True, num_classes: int = 2):
    try:
        # Newer torchvision API with Weights enums
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
    except Exception:
        # Fallback for older versions
        model = models.resnet50(pretrained=pretrained)
    in_feat = model.fc.in_features
    model.fc = nn.Linear(in_feat, num_classes)
    last_conv = 'layer4'
    return model, last_conv


def build_optimizer(model, lr=1e-4, weight_decay=1e-4):
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def train_one_epoch(model, loader, criterion, optimizer, device, accumulation_steps=1, use_amp=True, epoch=None, log_wandb=False, global_step=None):
    model.train()
    running_loss = 0.0
    all_targets, all_probs = [], []
    optimizer.zero_grad(set_to_none=True)
    
    # Mixed precision scaler
    scaler = GradScaler() if use_amp else None
    
    # Progress bar
    desc = f"Epoch {epoch} [Train]" if epoch is not None else "Training"
    pbar = tqdm(enumerate(loader), total=len(loader), desc=desc, leave=False)
    
    # Initialize global step if not provided
    if global_step is None:
        global_step = 0
    
    for batch_idx, (imgs, labels, _) in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Mixed precision forward pass
        if use_amp:
            with autocast():
                logits = model(imgs)
                loss = criterion(logits, labels)
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
        
        # Normalize loss for gradient accumulation
        loss = loss / accumulation_steps
        
        # Backward pass with mixed precision
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights every accumulation_steps batches
        if (batch_idx + 1) % accumulation_steps == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        running_loss += loss.item() * imgs.size(0) * accumulation_steps
        
        # Detach and move to CPU immediately to free GPU memory
        with torch.no_grad():
            if use_amp:
                # Compute probs in float32 for numerical stability
                probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
            else:
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_probs.extend(probs.tolist())
        all_targets.extend(labels.cpu().numpy().tolist())
        
        # Update progress bar with current loss
        current_loss = running_loss / ((batch_idx + 1) * imgs.size(0))
        pbar.set_postfix({'loss': f'{current_loss:.4f}'})
        
        # Log batch metrics to wandb every N steps
        if log_wandb and batch_idx % 10 == 0:
            try:
                import wandb
                wandb.log({
                    'batch/train_loss': current_loss,
                    'batch/step': global_step + batch_idx,
                }, step=global_step + batch_idx)
            except:
                pass
        
        # Free GPU memory explicitly
        del imgs, labels, logits, loss, probs
        
        # More aggressive memory cleanup for small GPUs
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
    
    pbar.close()
    
    # Handle remaining gradients if batches don't divide evenly
    if len(loader) % accumulation_steps != 0:
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    
    epoch_loss = running_loss / len(loader.dataset)
    try:
        epoch_auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        epoch_auc = float('nan')
    preds = (np.array(all_probs) >= 0.5).astype(int)
    epoch_acc = accuracy_score(all_targets, preds)
    
    # Return global step for next epoch
    return {'loss': epoch_loss, 'auc': epoch_auc, 'acc': epoch_acc, 'global_step': global_step + len(loader)}


def evaluate(model, loader, criterion, device, use_amp=True, epoch=None):
    model.eval()
    running_loss = 0.0
    all_targets, all_probs = [], []
    
    # Progress bar
    desc = f"Epoch {epoch} [Val]" if epoch is not None else "Evaluating"
    pbar = tqdm(enumerate(loader), total=len(loader), desc=desc, leave=False)
    
    with torch.no_grad():
        for batch_idx, (imgs, labels, _) in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Mixed precision forward pass
            if use_amp:
                with autocast():
                    logits = model(imgs)
                    loss = criterion(logits, labels)
            else:
                logits = model(imgs)
                loss = criterion(logits, labels)
            
            running_loss += loss.item() * imgs.size(0)
            
            if use_amp:
                probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
            else:
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            
            all_probs.extend(probs.tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
            
            # Update progress bar with current loss
            current_loss = running_loss / ((batch_idx + 1) * imgs.size(0))
            pbar.set_postfix({'loss': f'{current_loss:.4f}'})
            
            # Free GPU memory explicitly
            del imgs, labels, logits, loss, probs
            # More aggressive memory cleanup for small GPUs
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
    
    pbar.close()
    
    epoch_loss = running_loss / len(loader.dataset)
    try:
        epoch_auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        epoch_auc = float('nan')
    preds = (np.array(all_probs) >= 0.5).astype(int)
    epoch_acc = accuracy_score(all_targets, preds)
    cm = confusion_matrix(all_targets, preds)
    cls_report = classification_report(all_targets, preds, target_names=['HER2-', 'HER2+'], output_dict=True)
    return {
        'loss': epoch_loss,
        'auc': epoch_auc,
        'acc': epoch_acc,
        'cm': cm,
        'report': cls_report,
        'targets': all_targets,
        'probs': all_probs,
    }


def save_metrics_csv_json(history, best_metrics, logs_dir: Path):
    rows = []
    for e, rec in enumerate(history):
        row = {'epoch': e}
        row.update({f'train_{k}': rec['train'][k] for k in ['loss', 'auc', 'acc']})
        row.update({f'val_{k}': rec['val'][k] for k in ['loss', 'auc', 'acc']})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(logs_dir / 'metrics.csv', index=False)
    with open(logs_dir / 'best.json', 'w') as f:
        json.dump(best_metrics, f, indent=2)


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


def train_phase1(config: Dict[str, Any]):
    """Run Phase 1 training (ResNet-50) and save best model/logs.

    Args:
        config: dict containing training configuration. See module docstring for keys.

    Returns:
        dict with keys: best_model_path, best_metrics, logs_dir, models_dir
    """
    cfg = _with_defaults(config)

    set_seed(int(cfg.get('seed', 42)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    out_dir = Path(cfg['output_dir'])
    models_dir = out_dir / 'models'
    logs_dir = out_dir / 'logs'
    vis_dir = out_dir / 'gradcam'
    tb_dir = out_dir / 'tensorboard'
    for d in [out_dir, models_dir, logs_dir, vis_dir, tb_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=str(tb_dir))
    
    # Define checkpoint path and resume flag early
    checkpoint_path = models_dir / 'checkpoint_last.pth'
    resume_training = cfg.get('resume', True)
    
    # Initialize Weights & Biases
    use_wandb = cfg.get('use_wandb', True) and WANDB_AVAILABLE
    wandb_id = None
    if use_wandb:
        wandb_project = cfg.get('wandb_project', 'her2-classification')
        wandb_name = cfg.get('wandb_name', f"phase1_bs{cfg['batch_size']}_size{cfg['input_size']}")
        
        # Check if resuming and get wandb run ID from checkpoint
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
                settings=wandb.Settings(
                    code_dir=None,  # Disable code artifact logging
                    _disable_stats=False,
                    _disable_meta=False,
                )
            )
            wandb_id = wandb.run.id
            print(f"Weights & Biases initialized: {wandb_project}/{wandb_name}")
        except Exception as e:
            print(f"Warning: Failed to initialize Weights & Biases: {e}")
            print("Continuing with TensorBoard logging only")
            use_wandb = False
    else:
        if not WANDB_AVAILABLE:
            print("Weights & Biases not available - install with: pip install wandb")
        else:
            print("Weights & Biases logging disabled")

    # Datasets and loaders
    train_tfms = build_transforms(int(cfg['input_size']), train=True)
    val_tfms = build_transforms(int(cfg['input_size']), train=False)

    train_set = PatchDataset(cfg['train_csv'], cfg['path_col'], cfg['label_col'], transform=train_tfms)
    val_set = PatchDataset(cfg['val_csv'], cfg['path_col'], cfg['label_col'], transform=val_tfms)

    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg['batch_size']),
        shuffle=True,
        num_workers=int(cfg['num_workers']),
        pin_memory=pin_mem,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(cfg['batch_size']),
        shuffle=False,
        num_workers=int(cfg['num_workers']),
        pin_memory=pin_mem,
    )

    # Model, loss, optimizer
    model, last_conv = build_resnet50(cfg.get('pretrained', True))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, lr=float(cfg['lr']), weight_decay=float(cfg['weight_decay']))
    
    # Enable wandb model watching after model is created
    if use_wandb:
        wandb.watch(model, criterion, log='all', log_freq=100)
        print(f"Model watching enabled - tracking gradients and parameters")

    best_key = cfg.get('save_best_by', 'auc')
    if best_key not in ('auc', 'acc'):
        best_key = 'auc'
    best_score = -np.inf
    best_metrics = {}
    history = []
    start_epoch = 0
    
    # Check for checkpoint to resume from
    if resume_training and checkpoint_path.exists():
        print(f"Found checkpoint at {checkpoint_path}, resuming training...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_score = checkpoint['best_score']
            best_metrics = checkpoint.get('best_metrics', {})
            history = checkpoint.get('history', [])
            
            print(f"Resumed from epoch {checkpoint['epoch']}")
            print(f"Best {best_key} so far: {best_score:.4f}")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}")
            print("Starting training from scratch...")
            start_epoch = 0
    else:
        if resume_training:
            print("No checkpoint found, starting training from scratch...")
        else:
            print("Resume disabled, starting training from scratch...")
    
    # Get gradient accumulation steps
    accumulation_steps = int(cfg.get('accumulation_steps', 1))
    use_amp = cfg.get('use_amp', True) and torch.cuda.is_available()
    
    if accumulation_steps > 1:
        print(f"Using gradient accumulation with {accumulation_steps} steps (effective batch size: {int(cfg['batch_size']) * accumulation_steps})")
    if use_amp:
        print("Using mixed precision training (FP16) for memory efficiency")

    start_time = time.time()
    epochs = int(cfg['epochs'])
    
    # Track global training steps for batch-level logging
    global_step = 0
    
    # Main training loop with overall progress bar
    epoch_pbar = tqdm(range(start_epoch, epochs), desc="Training Progress", initial=start_epoch, total=epochs)
    
    for epoch in epoch_pbar:
        t0 = time.time()
        train_log = train_one_epoch(model, train_loader, criterion, optimizer, device, accumulation_steps, use_amp, epoch=epoch+1, log_wandb=use_wandb, global_step=global_step)
        global_step = train_log.get('global_step', global_step)  # Update global step
        val_log = evaluate(model, val_loader, criterion, device, use_amp, epoch=epoch+1)
        history.append({'train': {k: v for k, v in train_log.items() if k != 'global_step'}, 'val': {k: val_log[k] for k in ['loss', 'auc', 'acc']}})

        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_log['loss'], epoch)
        writer.add_scalar('Loss/val', val_log['loss'], epoch)
        writer.add_scalar('AUC/train', train_log['auc'], epoch)
        writer.add_scalar('AUC/val', val_log['auc'], epoch)
        writer.add_scalar('Accuracy/train', train_log['acc'], epoch)
        writer.add_scalar('Accuracy/val', val_log['acc'], epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Log to Weights & Biases
        if use_wandb:
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
            # Add GPU memory usage if available
            if torch.cuda.is_available():
                wandb_log['gpu/memory_allocated_gb'] = torch.cuda.memory_allocated() / 1024**3
                wandb_log['gpu/memory_reserved_gb'] = torch.cuda.memory_reserved() / 1024**3
                wandb_log['gpu/max_memory_allocated_gb'] = torch.cuda.max_memory_allocated() / 1024**3
            wandb.log(wandb_log)

        cur_score = val_log.get(best_key, float('-inf'))
        improved = np.isfinite(cur_score) and cur_score > best_score
        if improved:
            best_score = cur_score
            best_metrics = {
                'epoch': epoch,
                'score_key': best_key,
                'score': float(best_score),
                'val': {k: float(val_log[k]) if isinstance(val_log[k], (int, float, np.floating)) else None for k in ['loss', 'auc', 'acc']}
            }
            torch.save(model.state_dict(), models_dir / 'model_phase1.pth')
            
            # Log best metrics to wandb
            if use_wandb:
                wandb.run.summary['best_epoch'] = epoch
                wandb.run.summary['best_val_auc'] = float(val_log['auc'])
                wandb.run.summary['best_val_acc'] = float(val_log['acc'])
                wandb.run.summary['best_val_loss'] = float(val_log['loss'])
        
        dur = time.time() - t0
        
        # Update epoch progress bar with current metrics
        epoch_pbar.set_postfix({
            'train_auc': f'{train_log["auc"]:.4f}',
            'val_auc': f'{val_log["auc"]:.4f}',
            'best': '*' if improved else ''
        })
        
        # Print detailed epoch summary
        tqdm.write(
            f"Epoch {epoch+1}/{epochs} | "
            f"train: loss={train_log['loss']:.4f} auc={train_log['auc']:.4f} acc={train_log['acc']:.4f} | "
            f"val: loss={val_log['loss']:.4f} auc={val_log['auc']:.4f} acc={val_log['acc']:.4f} | "
            f"{'*' if improved else ''} ({dur:.1f}s)"
        )
        
        # Save checkpoint after each epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_score': best_score,
            'best_metrics': best_metrics,
            'history': history,
            'config': cfg,
            'wandb_id': wandb_id if use_wandb else None,
        }
        torch.save(checkpoint, models_dir / 'checkpoint_last.pth')
        
        # Save periodic checkpoints every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(checkpoint, models_dir / f'checkpoint_epoch_{epoch}.pth')
            tqdm.write(f"Saved checkpoint at epoch {epoch}")
        
        # Clean up after each epoch to prevent memory accumulation
        gc.collect()
        torch.cuda.empty_cache()
    
    epoch_pbar.close()
    
    # Close TensorBoard writer
    writer.close()

    # Save logs, confusion matrix, and report for final val run
    save_metrics_csv_json(history, best_metrics, logs_dir)
    # Use last validation metrics to write detailed CSVs
    try:
        cm = val_log['cm']
        rep = val_log['report']
        pd.DataFrame(cm, index=['HER2-', 'HER2+'], columns=['Pred -', 'Pred +']).to_csv(logs_dir / 'confusion_matrix.csv')
        pd.DataFrame(rep).to_csv(logs_dir / 'classification_report.csv')
        
        # Log confusion matrix to wandb
        if use_wandb:
            # Create wandb confusion matrix
            wandb.log({
                "conf_mat": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=val_log['targets'],
                    preds=(np.array(val_log['probs']) >= 0.5).astype(int),
                    class_names=['HER2-', 'HER2+']
                )
            })
            # Log classification report as table
            report_df = pd.DataFrame(rep).transpose()
            wandb.log({"classification_report": wandb.Table(dataframe=report_df)})
    except Exception as e:
        print(f"Warning: Failed to log final metrics: {e}")
    
    # Close wandb
    if use_wandb:
        wandb.finish()
        print("Weights & Biases run finished")

    best_model_path = models_dir / 'model_phase1.pth'
    print('Best model saved at:', best_model_path)
    print('Training finished in', f"{(time.time()-start_time)/60:.1f} min")
    print(f'TensorBoard logs saved to: {tb_dir}')
    print(f'To view: tensorboard --logdir={tb_dir}')
    if use_wandb:
        print(f'Weights & Biases dashboard: {wandb.run.url}')

    return {
        'best_model_path': str(best_model_path),
        'best_metrics': best_metrics,
        'logs_dir': str(logs_dir),
        'models_dir': str(models_dir),
        'tb_dir': str(tb_dir),
        'wandb_url': wandb.run.url if use_wandb else None,
    }
