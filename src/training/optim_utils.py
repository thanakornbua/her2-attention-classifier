"""
Optimizer and learning rate scheduling utilities.
"""

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from typing import Dict, Any


def build_optimizer(
    model,
    optimizer_name: str = 'adam',
    lr: float = 1e-4,
    weight_decay: float = 1e-5
) -> optim.Optimizer:
    """
    Build optimizer for model.
    
    Args:
        model: PyTorch model
        optimizer_name: 'adam' or 'sgd'
        lr: Learning rate
        weight_decay: L2 regularization
        
    Returns:
        Optimizer instance
    """
    if optimizer_name.lower() == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def build_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str = 'reduce_lr',
    config: Dict[str, Any] = None
):
    """
    Build learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: 'reduce_lr' or 'cosine'
        config: Scheduler configuration
        
    Returns:
        Learning rate scheduler
    """
    if config is None:
        config = {}
    
    if scheduler_type == 'reduce_lr':
        return ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=config.get('factor', 0.1),
            patience=config.get('patience', 5),
            verbose=True
        )
    elif scheduler_type == 'cosine':
        return CosineAnnealingLR(
            optimizer,
            T_max=config.get('T_max', 100),
            eta_min=config.get('eta_min', 1e-6)
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
