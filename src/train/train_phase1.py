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
import torchvision.transforms as T
import torchvision.models as models

from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report

# Stain normalization (Macenko)
from src.preprocessing.stain_normalization import macenko_normalization


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
        if self.transform:
            img = self.transform(img)
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


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_targets, all_probs = [], []
    for imgs, labels, _ in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        probs = torch.softmax(logits.detach(), dim=1)[:, 1].cpu().numpy()
        all_probs.extend(probs.tolist())
        all_targets.extend(labels.cpu().numpy().tolist())
    epoch_loss = running_loss / len(loader.dataset)
    try:
        epoch_auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        epoch_auc = float('nan')
    preds = (np.array(all_probs) >= 0.5).astype(int)
    epoch_acc = accuracy_score(all_targets, preds)
    return {'loss': epoch_loss, 'auc': epoch_auc, 'acc': epoch_acc}


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets, all_probs = [], []
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            running_loss += loss.item() * imgs.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
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
    for d in [out_dir, models_dir, logs_dir, vis_dir]:
        d.mkdir(parents=True, exist_ok=True)

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

    best_key = cfg.get('save_best_by', 'auc')
    if best_key not in ('auc', 'acc'):
        best_key = 'auc'
    best_score = -np.inf
    best_metrics = {}
    history = []

    start_time = time.time()
    epochs = int(cfg['epochs'])
    for epoch in range(epochs):
        t0 = time.time()
        train_log = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_log = evaluate(model, val_loader, criterion, device)
        history.append({'train': train_log, 'val': {k: val_log[k] for k in ['loss', 'auc', 'acc']}})

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
        dur = time.time() - t0
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"train: loss={train_log['loss']:.4f} auc={train_log['auc']:.4f} acc={train_log['acc']:.4f} | "
            f"val: loss={val_log['loss']:.4f} auc={val_log['auc']:.4f} acc={val_log['acc']:.4f} | "
            f"{'*' if improved else ''} ({dur:.1f}s)"
        )

    # Save logs, confusion matrix, and report for final val run
    save_metrics_csv_json(history, best_metrics, logs_dir)
    # Use last validation metrics to write detailed CSVs
    try:
        cm = val_log['cm']
        rep = val_log['report']
        pd.DataFrame(cm, index=['HER2-', 'HER2+'], columns=['Pred -', 'Pred +']).to_csv(logs_dir / 'confusion_matrix.csv')
        pd.DataFrame(rep).to_csv(logs_dir / 'classification_report.csv')
    except Exception:
        pass

    best_model_path = models_dir / 'model_phase1.pth'
    print('Best model saved at:', best_model_path)
    print('Training finished in', f"{(time.time()-start_time)/60:.1f} min")

    return {
        'best_model_path': str(best_model_path),
        'best_metrics': best_metrics,
        'logs_dir': str(logs_dir),
        'models_dir': str(models_dir),
    }
