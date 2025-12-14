"""
Grad-CAM and attention visualization for model explainability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for CNN explainability.
    
    References:
        Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks 
        via Gradient-based Localization" (ICCV 2017)
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model: CNN model
            target_layer: Layer to compute CAM from (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """Hook to save forward activations."""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward gradients."""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Class Activation Map.
        
        Args:
            input_tensor: Input image tensor [1, C, H, W]
            target_class: Target class for CAM (None = predicted class)
            
        Returns:
            CAM heatmap [H, W] normalized to [0, 1]
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
        
        # Compute CAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2), keepdim=True)  # [C, 1, 1]
        
        # Weighted sum of activations
        cam = (weights * activations).sum(dim=0)  # [H, W]
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def remove_hooks(self):
        """Remove registered hooks."""
        self.forward_hook.remove()
        self.backward_hook.remove()


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay heatmap on original image.
    
    Args:
        image: Original image [H, W, 3] in range [0, 255]
        heatmap: CAM heatmap [H, W] in range [0, 1]
        alpha: Blending factor
        colormap: OpenCV colormap
        
    Returns:
        Overlaid image [H, W, 3]
    """
    # Resize heatmap to image size
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Convert heatmap to RGB
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend with original image
    overlaid = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlaid


def visualize_attention_weights(
    attention_weights: np.ndarray,
    patch_coords: np.ndarray,
    slide_shape: Tuple[int, int],
    output_path: Optional[Path] = None,
    patch_size: int = 512
) -> np.ndarray:
    """
    Visualize MIL attention weights as a heatmap on slide.
    
    Args:
        attention_weights: Attention scores per patch [num_patches]
        patch_coords: Patch coordinates [num_patches, 2] (x, y)
        slide_shape: Slide dimensions (width, height)
        output_path: Optional path to save visualization
        patch_size: Size of each patch
        
    Returns:
        Attention heatmap [height, width]
    """
    width, height = slide_shape
    
    # Create empty heatmap
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    # Fill in attention weights for each patch
    for i, (x, y) in enumerate(patch_coords):
        x_end = min(x + patch_size, width)
        y_end = min(y + patch_size, height)
        
        heatmap[y:y_end, x:x_end] = attention_weights[i]
    
    # Normalize
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # Visualize
    if output_path is not None:
        plt.figure(figsize=(12, 8))
        plt.imshow(heatmap, cmap='jet', interpolation='bilinear')
        plt.colorbar(label='Attention Weight')
        plt.title('MIL Attention Heatmap')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return heatmap


def generate_gradcam_for_patches(
    model: nn.Module,
    patches: torch.Tensor,
    target_class: Optional[int] = None,
    device: torch.device = torch.device('cpu')
) -> np.ndarray:
    """
    Generate Grad-CAM for a batch of patches.
    
    Args:
        model: Trained patch classifier
        patches: Patch tensor [batch_size, C, H, W]
        target_class: Target class (None = predicted class)
        device: Device to use
        
    Returns:
        Array of CAM heatmaps [batch_size, H, W]
    """
    model = model.to(device)
    model.eval()
    
    # Get last convolutional layer
    if hasattr(model, 'backbone'):
        # For PatchClassifier
        if hasattr(model.backbone, '__getitem__'):
            target_layer = model.backbone[-1]
        else:
            # For sequential backbone
            target_layer = list(model.backbone.children())[-1]
    else:
        raise ValueError("Model structure not supported for Grad-CAM")
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model, target_layer)
    
    cams = []
    for i in range(patches.size(0)):
        patch = patches[i:i+1].to(device)
        cam = gradcam.generate_cam(patch, target_class)
        cams.append(cam)
    
    gradcam.remove_hooks()
    
    return np.stack(cams)
