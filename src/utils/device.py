"""
Device detection and management utilities.
"""

import torch


def get_device(force_cpu: bool = False) -> torch.device:
    """
    Get the best available device (CUDA if available, else CPU).
    
    Args:
        force_cpu (bool): Force CPU usage even if CUDA is available
        
    Returns:
        torch.device: Device to use for training
    """
    if force_cpu:
        return torch.device('cpu')
    
    if torch.cuda.is_available():
        return torch.device('cuda')
    
    return torch.device('cpu')


def print_device_info(device: torch.device):
    """
    Print information about the selected device.
    
    Args:
        device (torch.device): Device to print info about
    """
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"  CUDA Version: {torch.version.cuda}")
