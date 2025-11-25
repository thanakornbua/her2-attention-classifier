#!/usr/bin/env python3
"""Phase 1 training entrypoint that consumes Zarr manifests directly.

This script mirrors the notebook pipeline so it can be launched from shell
scripts (e.g. run_preprocessing.sh) and higher-level orchestration.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as T
import zarr
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.cuda import amp
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover - wandb is optional
    wandb = None
    WANDB_AVAILABLE = False

from torchvision.transforms import functional as TF


class RandomElasticDeformation:
    """Applies a lightweight elastic deformation to PIL images."""

    def __init__(self, alpha: float = 2.0, sigma: float = 5.0, p: float = 0.5, kernel_size: int = 15) -> None:
        self.alpha = alpha
        self.sigma = sigma
        self.p = p
        self.kernel_size = kernel_size | 1  # ensure odd for blur

    def __call__(self, img):
        if random.random() > self.p:
            return img
        tensor = TF.to_tensor(img).unsqueeze(0)
        _, _, h, w = tensor.shape
        device = tensor.device
        with torch.no_grad():
            noise = torch.randn(1, 2, h, w, device=device)
            blurred = TF.gaussian_blur(
                noise,
                kernel_size=[self.kernel_size, self.kernel_size],
                sigma=(self.sigma, self.sigma),
            )
            flow = blurred * (self.alpha / max(h, w))
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(-1, 1, h, device=device),
                torch.linspace(-1, 1, w, device=device),
                indexing="ij",
            )
            base_grid = torch.stack((grid_x, grid_y), dim=-1)
            grid = base_grid + flow.squeeze(0).permute(1, 2, 0)
            grid = grid.unsqueeze(0)
            warped = F.grid_sample(tensor, grid, mode="bilinear", padding_mode="reflection", align_corners=True)
        warped = warped.squeeze(0).clamp(0.0, 1.0)
        return TF.to_pil_image(warped)


class SimpleZarrPatchDataset(Dataset):
    """Patch-level dataset backed by a Zarr manifest."""

    def __init__(
        self,
        manifest_path: Path,
        transform=None,
        max_patches_per_slide: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self._metadata = pd.read_csv(self.manifest_path)
        required_cols = {"zarr_path", "label", "num_patches"}
        missing = required_cols.difference(self._metadata.columns)
        if missing:
            raise ValueError(f"Manifest missing columns: {missing}")

        self.transform = transform
        self.max_patches_per_slide = max_patches_per_slide
        self.rng = np.random.default_rng(seed)
        self._zarr_cache: Dict[str, zarr.hierarchy.Group] = {}
        self.num_classes = int(self._metadata["label"].nunique())
        self.indices = self._build_index()

    def _build_index(self) -> List[Tuple[int, int]]:
        indices: List[Tuple[int, int]] = []
        for row_idx, row in self._metadata.iterrows():
            total = int(row["num_patches"])
            if total <= 0:
                continue
            if self.max_patches_per_slide is None or total <= self.max_patches_per_slide:
                chosen = np.arange(total)
            else:
                chosen = self.rng.choice(total, size=self.max_patches_per_slide, replace=False)
                chosen = np.sort(chosen)
            for local_idx in chosen:
                indices.append((row_idx, int(local_idx)))
        if not indices:
            raise ValueError(f"No patches indexed from {self.manifest_path}")
        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def _get_zarr(self, path: str) -> zarr.hierarchy.Group:
        if path not in self._zarr_cache:
            self._zarr_cache[path] = zarr.open(path, mode="r")
        return self._zarr_cache[path]

    def __getitem__(self, idx: int):
        row_idx, local_idx = self.indices[idx]
        row = self._metadata.iloc[row_idx]
        g = self._get_zarr(row["zarr_path"])
        patch = g["patches"][local_idx]
        # Copy to avoid holding reference to zarr array
        img = Image.fromarray(np.array(patch, copy=True))
        if self.transform is not None:
            img = self.transform(img)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return img, label
    
    def __del__(self):
        """Explicitly close zarr handles on cleanup."""
        if hasattr(self, '_zarr_cache'):
            self._zarr_cache.clear()


def build_transforms(
    rotation_degrees: float,
    elastic_params: Optional[Dict[str, float]],
) -> Tuple[T.Compose, T.Compose]:
    elastic = (
        RandomElasticDeformation(
            alpha=elastic_params.get("alpha", 2.0),
            sigma=elastic_params.get("sigma", 5.0),
            p=elastic_params.get("prob", 0.6),
            kernel_size=int(elastic_params.get("kernel_size", 15)),
        )
        if elastic_params is not None
        else None
    )
    train_ops = [T.RandomHorizontalFlip(), T.RandomVerticalFlip()]
    if rotation_degrees > 0:
        train_ops.append(T.RandomRotation(degrees=rotation_degrees))
    if elastic is not None:
        train_ops.append(elastic)
    train_ops.extend(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_ops = [
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return T.Compose(train_ops), T.Compose(val_ops)


def prepare_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(name: str, num_classes: int, dropout_p: float = 0.0) -> torch.nn.Module:
    if name == "resnet50":
        model = tv_models.resnet50(weights=None)
        in_features = model.fc.in_features
        if dropout_p > 0:
            model.fc = nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(in_features, num_classes)
            )
        else:
            model.fc = nn.Linear(in_features, num_classes)
    elif name == "efficientnet_b0":
        model = tv_models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        if dropout_p > 0:
            model.classifier[1] = nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(in_features, num_classes)
            )
        else:
            model.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model '{name}'")
    return model


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    model_name,
    writer,
    wandb_run,
    epoch,
    max_steps=None,
    scheduler=None,
):
    model.train()
    losses = []
    total_steps = len(loader) if max_steps is None else min(len(loader), max_steps)
    progress = tqdm(loader, total=total_steps, desc=f"{model_name} train", leave=False)
    base_epoch = epoch - 1
    for step, (images, labels) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step(base_epoch + step / max(1, len(loader)))
        
        # Store loss value immediately and release tensor
        loss_value = loss.detach().item()
        losses.append(loss_value)
        del loss, logits  # Explicit cleanup
        
        if writer is not None:
            writer.add_scalar(f"{model_name}/train_loss_step", loss_value, (epoch - 1) * len(loader) + step)
        if wandb_run is not None:
            wandb_run.log({"train/loss": loss_value, "step": (epoch - 1) * len(loader) + step}, commit=False)
        progress.set_postfix(loss=f"{loss_value:.4f}")
        
        # Periodic cleanup
        if step % 100 == 0:
            torch.cuda.empty_cache() if device.type == "cuda" else None
        
        if step >= total_steps:
            break
    progress.close()
    return float(np.mean(losses)) if losses else math.nan


def evaluate(model, loader, criterion, device, model_name, max_steps=None, compute_auc=False):
    model.eval()
    losses = []
    correct = 0
    total = 0
    all_probs = []
    all_labels = []
    total_steps = len(loader) if max_steps is None else min(len(loader), max_steps)
    progress = tqdm(loader, total=total_steps, desc=f"{model_name} val", leave=False)
    with torch.no_grad():
        for step, (images, labels) in enumerate(progress, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            
            # Compute metrics and immediately extract values
            loss_value = loss.item()
            losses.append(loss_value)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            if compute_auc:
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
            
            # Explicit cleanup
            del logits, loss, preds
            if compute_auc:
                del probs
            
            progress.set_postfix(loss=f"{loss_value:.4f}")
            
            # Periodic cleanup
            if step % 50 == 0:
                torch.cuda.empty_cache() if device.type == "cuda" else None
            
            if step >= total_steps:
                break
    progress.close()
    avg_loss = float(np.mean(losses)) if losses else math.nan
    acc = correct / total if total else math.nan
    
    auc = None
    if compute_auc and all_probs:
        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        try:
            if all_probs.shape[1] == 2:
                # Binary classification
                auc = roc_auc_score(all_labels, all_probs[:, 1])
            else:
                # Multi-class
                auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        except ValueError:
            auc = None
    
    return avg_loss, acc, auc


def run_training(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "tensorboard"
    log_dir.mkdir(parents=True, exist_ok=True)

    elastic_cfg = None if args.disable_elastic else {
        "alpha": args.elastic_alpha,
        "sigma": args.elastic_sigma,
        "prob": args.elastic_prob,
        "kernel_size": args.elastic_kernel_size,
    }
    rotation_degrees = 0.0 if args.disable_rotation else args.rotation_degrees
    train_tfms, val_tfms = build_transforms(rotation_degrees, elastic_cfg)
    train_ds = SimpleZarrPatchDataset(
        args.train_manifest,
        transform=train_tfms,
        max_patches_per_slide=args.max_patches_per_slide,
        seed=args.seed,
    )
    val_ds = SimpleZarrPatchDataset(
        args.val_manifest,
        transform=val_tfms,
        max_patches_per_slide=args.max_patches_per_slide,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    
    # Optional test set
    test_loader = None
    if args.test_manifest and Path(args.test_manifest).exists():
        test_ds = SimpleZarrPatchDataset(
            args.test_manifest,
            transform=val_tfms,
            max_patches_per_slide=args.max_patches_per_slide,
            seed=args.seed,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True if args.num_workers > 0 else False,
        )

    device = prepare_device(args.device)
    model = build_model(args.model, train_ds.num_classes, dropout_p=args.dropout_p).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.adamw_beta1, args.adamw_beta2),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.scheduler_t0,
        T_mult=args.scheduler_t_mult,
        eta_min=args.scheduler_eta_min,
    )
    scaler = amp.GradScaler(enabled=device.type == "cuda")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_writer = SummaryWriter(log_dir=str(log_dir / f"{args.model}_{timestamp}")) if args.enable_tensorboard else None

    wandb_run = None
    if args.enable_wandb and WANDB_AVAILABLE:
        run_name = args.wandb_run_name or f"{args.model}-{timestamp}"
        wandb_config = vars(args).copy()
        wandb_config.pop("wandb_api_key", None)
        wandb_run = wandb.init(project=args.wandb_project, name=run_name, config=wandb_config)
        wandb_run.watch(model, log="gradients", log_freq=50)

    best_auc = -float("inf")
    best_acc = -float("inf")
    best_state = None
    history = []
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            args.model,
            tb_writer,
            wandb_run,
            epoch,
            max_steps=args.train_steps_per_epoch,
            scheduler=scheduler,
        )
        val_loss, val_acc, val_auc = evaluate(
            model,
            val_loader,
            criterion,
            device,
            args.model,
            max_steps=args.val_steps,
            compute_auc=True,
        )
        
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_auc": val_auc if val_auc is not None else 0.0,
        }
        history.append(epoch_record)
        
        if tb_writer is not None:
            tb_writer.add_scalar(f"{args.model}/train_loss_epoch", train_loss, epoch)
            tb_writer.add_scalar(f"{args.model}/val_loss", val_loss, epoch)
            tb_writer.add_scalar(f"{args.model}/val_acc", val_acc, epoch)
            if val_auc is not None:
                tb_writer.add_scalar(f"{args.model}/val_auc", val_auc, epoch)
        
        log_dict = {
            "epoch": epoch,
            "train/loss_epoch": train_loss,
            "val/loss": val_loss,
            "val/acc": val_acc,
        }
        if val_auc is not None:
            log_dict["val/auc"] = val_auc
        if wandb_run is not None:
            wandb_run.log(log_dict)
        
        # Early stopping based on AUC
        current_metric = val_auc if val_auc is not None else val_acc
        if current_metric > best_auc:
            best_auc = current_metric
            best_acc = val_acc
            best_state = model.state_dict()
            torch.save(best_state, output_dir / f"best_{args.model}.pth")
            epochs_without_improvement = 0
            print(f"✓ Epoch {epoch}: New best AUC={val_auc:.4f} (acc={val_acc:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  Epoch {epoch}: AUC={val_auc:.4f}, acc={val_acc:.4f} (no improvement for {epochs_without_improvement} epochs)")
        
        # Early stopping check
        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(f"\n⚠ Early stopping triggered after {epoch} epochs (no AUC improvement for {args.early_stop_patience} epochs)")
            break
        
        # End-of-epoch cleanup
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # Evaluate on test set if available
    test_results = None
    if test_loader is not None and best_state is not None:
        print("\n" + "="*50)
        print("Evaluating on test set with best model...")
        print("="*50)
        model.load_state_dict(best_state)
        test_loss, test_acc, test_auc = evaluate(
            model,
            test_loader,
            criterion,
            device,
            args.model,
            max_steps=None,
            compute_auc=True,
        )
        test_results = {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_auc": test_auc if test_auc is not None else 0.0,
        }
        print(f"Test Results: loss={test_loss:.4f}, acc={test_acc:.4f}, auc={test_auc:.4f}")
        
        if tb_writer is not None:
            tb_writer.add_scalar(f"{args.model}/test_loss", test_loss, args.epochs)
            tb_writer.add_scalar(f"{args.model}/test_acc", test_acc, args.epochs)
            if test_auc is not None:
                tb_writer.add_scalar(f"{args.model}/test_auc", test_auc, args.epochs)
        
        if wandb_run is not None:
            wandb_run.log({"test/loss": test_loss, "test/acc": test_acc, "test/auc": test_auc if test_auc is not None else 0.0})
    
    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()
    if wandb_run is not None:
        wandb_run.finish()

    summary = {
        "model": args.model,
        "best_val_auc": best_auc,
        "best_val_acc": best_acc,
        "history": history,
        "output_dir": str(output_dir),
    }
    if test_results:
        summary["test_results"] = test_results
    
    with open(output_dir / f"summary_{args.model}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nTraining complete:", json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1 training on Zarr manifests")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, default=None, help="Optional test manifest for hold-out evaluation")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase1_training"))
    parser.add_argument("--model", choices=["resnet50", "efficientnet_b0"], default="resnet50")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--adamw-beta1", type=float, default=0.9)
    parser.add_argument("--adamw-beta2", type=float, default=0.999)
    parser.add_argument("--scheduler-t0", type=int, default=2)
    parser.add_argument("--scheduler-t-mult", type=int, default=2)
    parser.add_argument("--scheduler-eta-min", type=float, default=1e-6)
    parser.add_argument("--max-patches-per-slide", type=int, default=2048)
    parser.add_argument("--train-steps-per-epoch", type=int, default=200)
    parser.add_argument("--val-steps", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-tensorboard", action="store_true")
    parser.add_argument("--enable-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="her2-phase1-cli")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--rotation-degrees", type=float, default=15.0)
    parser.add_argument("--disable-rotation", action="store_true")
    parser.add_argument("--elastic-alpha", type=float, default=2.0)
    parser.add_argument("--elastic-sigma", type=float, default=5.0)
    parser.add_argument("--elastic-prob", type=float, default=0.6)
    parser.add_argument("--elastic-kernel-size", type=int, default=15)
    parser.add_argument("--disable-elastic", action="store_true")
    parser.add_argument("--dropout-p", type=float, default=0.0, help="Dropout probability (0.3-0.5 recommended)")
    parser.add_argument("--early-stop-patience", type=int, default=0, help="Stop if AUC doesn't improve for N epochs (0=disabled)")
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
