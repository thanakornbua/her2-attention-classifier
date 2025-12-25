import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import logging

from src.models.patch_classifier import PatchClassifier
from src.dataloader.zarr_patch_dataset import ZarrPatchDataset

logger = logging.getLogger(__name__)

def extract_features_to_disk(
    patch_metadata: pd.DataFrame,
    model_path: Path,
    output_dir: Path,
    batch_size: int = 32,
    device: str = 'cuda',
    backbone_name: str = 'efficientnet_b0'
):
    """
    Extracts features from all patches using a trained Phase 1 model and saves them 
    as .npy files per slide (bag) for Phase 2 MIL training.
    
    Args:
        patch_metadata (pd.DataFrame): Metadata containing patch locations and slide IDs.
        model_path (Path): Path to the trained Phase 1 model checkpoint.
        output_dir (Path): Directory to save the extracted feature .npy files.
        batch_size (int): Batch size for inference.
        device (str): Device to run inference on ('cuda' or 'cpu').
        backbone_name (str): Name of the backbone architecture.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Model
    logger.info(f"Loading Phase 1 model from {model_path} (Backbone: {backbone_name})")
    model = PatchClassifier(backbone_name=backbone_name, num_classes=2, pretrained=False)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
        
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle state dict keys if they start with 'module.' or 'model.'
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '').replace('model.', '')
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict, strict=False)
    
    model.to(device)
    model.eval()
    
    # 2. Setup Dataset & Loader
    # We want to extract features for ALL patches in the metadata
    dataset = ZarrPatchDataset(
        zarr_root=None, # Deprecated/handled by metadata routing
        indices=range(len(patch_metadata)),
        patch_metadata=patch_metadata,
        return_metadata=True # Returns (img, label, loc, slide_name, case_name)
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=8, 
        pin_memory=True
    )
    
    # 3. Extraction Loop
    logger.info(f"Starting feature extraction for {len(dataset)} patches...")
    
    # Buffer to hold features for current slide
    # slide_name -> list of feature arrays
    features_buffer = {} 
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting Features"):
            imgs, labels, _, slide_names, case_names = batch
            imgs = imgs.to(device)
            
            # Extract features (before classification head)
            if hasattr(model.backbone, 'forward_features'):
                feats = model.backbone.forward_features(imgs)
            else:
                # Fallback for standard ResNet or other backbones
                feats = model.backbone(imgs) 
            
            # Global Average Pooling if output is spatial [B, C, H, W]
            if len(feats.shape) == 4:
                feats = torch.mean(feats, dim=(2, 3)) # [B, C]
            
            feats = feats.cpu().numpy()
            
            # Group by slide
            for i, slide_name in enumerate(slide_names):
                if slide_name not in features_buffer:
                    features_buffer[slide_name] = []
                features_buffer[slide_name].append(feats[i])
    
    # 4. Save to Disk
    logger.info(f"Saving features for {len(features_buffer)} slides to {output_dir}...")
    for slide_name, feats_list in tqdm(features_buffer.items(), desc="Saving .npy"):
        save_path = output_dir / f"{slide_name}.npy"
        feats_array = np.vstack(feats_list)
        np.save(save_path, feats_array)
        
    logger.info("Feature extraction complete.")
