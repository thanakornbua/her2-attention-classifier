import torch
from torch.optim import Optimizer

from .losses import compute_multi_loss


def set_model_phase(model: torch.nn.Module, phase_1: bool = True):
    """Freeze backbone in phase 1, unfreeze in phase 2."""
    # assume model.backbone exists
    for p in model.backbone.parameters():
        p.requires_grad = not phase_1


def train_epoch(model: torch.nn.Module,
                dataloader,
                optimizer: Optimizer,
                class_weights: torch.Tensor,
                phase_1: bool,
                device: torch.device) -> float:
    model.train()
    set_model_phase(model, phase_1)

    total_loss = 0.0
    n = 0

    for imgs, cls_targets, loc_targets in dataloader:
        imgs = imgs.to(device)
        cls_targets = cls_targets.to(device)
        loc_targets = loc_targets.to(device)
        class_weights = class_weights.to(device)

        optimizer.zero_grad(set_to_none=True)
        cls_logits, loc_out = model(imgs)
        loss = compute_multi_loss(cls_logits, loc_out, cls_targets, loc_targets, class_weights)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)

    return total_loss / max(n, 1)
