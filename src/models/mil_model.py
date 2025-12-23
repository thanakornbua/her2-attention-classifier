"""
Attention-based Multiple Instance Learning (MIL) model for slide-level classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMIL(nn.Module):
    """
    Attention-based MIL model for slide-level prediction.
    
    Architecture:
        - Feature extraction (frozen backbone from Phase 1)
        - Attention mechanism (learn instance importance)
        - Aggregation (weighted sum of instances)
        - Classification head
    
    References:
        Ilse et al. "Attention-based Deep Multiple Instance Learning" (ICML 2018)
    """
    
    def __init__(
        self,
        feature_dim: int = 2048,
        hidden_dim: int = 512,
        num_classes: int = 2,
        dropout: float = 0.25
    ):
        """
        Args:
            feature_dim: Dimension of input features (from backbone)
            hidden_dim: Hidden dimension for attention network
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()
        
        # Attention network
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, features, roi_confidence=None, return_attention=False):
        """
        Forward pass.
        
        Args:
            features: Patch features [batch_size, num_patches, feature_dim]
            roi_confidence: Optional ROI confidence weights [batch_size, num_patches]
            return_attention: Whether to return attention weights
            
        Returns:
            logits: Class logits [batch_size, num_classes]
            attention_weights (optional): Attention weights [batch_size, num_patches]
        """
        # Compute attention scores
        attention_scores = self.attention(features)  # [B, N, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [B, N, 1]
        
        # Apply ROI confidence weighting if provided
        if roi_confidence is not None:
            # roi_confidence shape: [B, N] -> [B, N, 1]
            roi_confidence_expanded = roi_confidence.unsqueeze(-1)
            # Weight attention by ROI confidence and renormalize
            attention_weights = attention_weights * roi_confidence_expanded
            attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # Aggregate features with attention
        slide_features = torch.sum(
            features * attention_weights, dim=1
        )  # [B, feature_dim]
        
        # Classify
        logits = self.classifier(slide_features)  # [B, num_classes]
        
        if return_attention:
            return logits, attention_weights.squeeze(-1)
        
        return logits


class GatedAttentionMIL(nn.Module):
    """
    Gated attention MIL model with more sophisticated attention mechanism.
    
    Uses gating to allow the model to learn which features to attend to.
    """
    
    def __init__(
        self,
        feature_dim: int = 2048,
        hidden_dim: int = 512,
        num_classes: int = 2,
        dropout: float = 0.25
    ):
        """
        Args:
            feature_dim: Dimension of input features
            hidden_dim: Hidden dimension for attention
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()
        
        # Attention network with gating
        self.attention_V = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh()
        )
        
        self.attention_U = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.attention_weights = nn.Linear(hidden_dim, 1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, features, roi_confidence=None, return_attention=False):
        """
        Forward pass with gated attention.
        
        Args:
            features: Patch features [batch_size, num_patches, feature_dim]
            roi_confidence: Optional ROI confidence weights [batch_size, num_patches]
            return_attention: Whether to return attention weights
            
        Returns:
            logits: Class logits [batch_size, num_classes]
            attention_weights (optional): Attention weights [batch_size, num_patches]
        """
        # Gated attention
        attention_V = self.attention_V(features)  # [B, N, hidden_dim]
        attention_U = self.attention_U(features)  # [B, N, hidden_dim]
        
        # Element-wise multiplication (gating)
        attention_gated = attention_V * attention_U  # [B, N, hidden_dim]
        
        # Compute attention scores
        attention_scores = self.attention_weights(attention_gated)  # [B, N, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [B, N, 1]
        
        # Apply ROI confidence weighting if provided
        if roi_confidence is not None:
            # roi_confidence shape: [B, N] -> [B, N, 1]
            roi_confidence_expanded = roi_confidence.unsqueeze(-1)
            # Weight attention by ROI confidence and renormalize
            attention_weights = attention_weights * roi_confidence_expanded
            attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # Aggregate features
        slide_features = torch.sum(
            features * attention_weights, dim=1
        )  # [B, feature_dim]
        
        # Classify
        logits = self.classifier(slide_features)  # [B, num_classes]
        
        if return_attention:
            return logits, attention_weights.squeeze(-1)
        
        return logits
