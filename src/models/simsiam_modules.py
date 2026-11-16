# coding: utf-8
# SimSiam Components for BM3 Integration
# Based on "Exploring Simple Siamese Representation Learning" (Chen & He, 2020)
# Reference: https://github.com/facebookresearch/simsiam

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimSiamEncoder(nn.Module):
    """
    SimSiam Encoder (3-layer projector/MLP).
    
    Projects input embeddings to a higher-dimensional representation space
    using a 3-layer MLP with BatchNorm and ReLU activations.
    
    Architecture:
        Input → Linear → BatchNorm → ReLU 
              → Linear → BatchNorm → ReLU
              → Linear → BatchNorm (no affine, no ReLU)
    
    Args:
        input_dim (int): Dimension of input embeddings
        hidden_dim (int): Dimension of hidden layers (default: 256)
        output_dim (int): Dimension of output embeddings (default: 128)
    
    Example:
        >>> encoder = SimSiamEncoder(input_dim=128, hidden_dim=256, output_dim=128)
        >>> x = torch.randn(32, 128)  # batch_size=32, input_dim=128
        >>> z = encoder(x)
        >>> print(z.shape)  # torch.Size([32, 128])
    """
    
    def __init__(self, input_dim, hidden_dim=256, output_dim=128):
        super(SimSiamEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # First layer: input_dim → hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        # Second layer: hidden_dim → hidden_dim
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Third layer (output): hidden_dim → output_dim
        self.fc3 = nn.Linear(hidden_dim, output_dim, bias=False)
        # Output BatchNorm without affine parameters (no learnable scale/shift)
        self.bn3 = nn.BatchNorm1d(output_dim, affine=False)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
    
    def forward(self, x):
        """
        Forward pass through the encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
        
        Returns:
            torch.Tensor: Encoded tensor of shape (batch_size, output_dim)
        """
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        
        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        
        # Layer 3 (output)
        x = self.fc3(x)
        x = self.bn3(x)
        
        return x


class SimSiamPredictor(nn.Module):
    """
    SimSiam Predictor (2-layer MLP).
    
    Transforms encoder outputs using a 2-layer MLP with BatchNorm and ReLU.
    This asymmetry between encoder and predictor is key to SimSiam's success.
    
    Architecture:
        Input → Linear → BatchNorm → ReLU → Linear
    
    Args:
        input_dim (int): Dimension of input embeddings (typically matches encoder output)
        hidden_dim (int): Dimension of hidden layer (default: 64)
        output_dim (int): Dimension of output embeddings (default: 128, matches encoder output)
    
    Example:
        >>> predictor = SimSiamPredictor(input_dim=128, hidden_dim=64, output_dim=128)
        >>> z = torch.randn(32, 128)  # encoded features
        >>> p = predictor(z)
        >>> print(p.shape)  # torch.Size([32, 128])
    """
    
    def __init__(self, input_dim, hidden_dim=64, output_dim=128):
        super(SimSiamPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # First layer: input_dim → hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        # Second layer (output): hidden_dim → output_dim
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=True)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x):
        """
        Forward pass through the predictor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
        
        Returns:
            torch.Tensor: Predicted tensor of shape (batch_size, output_dim)
        """
        # Layer 1 with activation
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        
        # Layer 2 (output, no activation)
        x = self.fc2(x)
        
        return x


class MultiModalFusion(nn.Module):
    """
    Multi-Modal Feature Fusion Layer.
    
    Fuses text and image features by:
    1. Projecting each modality to a common dimension (feat_embed_dim)
    2. Concatenating the projected features
    3. Optionally projecting the concatenated features to a target dimension
    
    Handles cases where only one modality is available gracefully.
    
    Args:
        text_feat_dim (int): Dimension of input text features (e.g., 384)
        image_feat_dim (int): Dimension of input image features (e.g., 4096)
        feat_embed_dim (int): Target dimension for each modality projection (default: 64)
        output_dim (int, optional): If provided, concatenated features are projected to this dimension.
                                    If None, output is concatenation of text and image projections.
    
    Example:
        >>> fusion = MultiModalFusion(text_feat_dim=384, image_feat_dim=4096, feat_embed_dim=64)
        >>> text_feat = torch.randn(7050, 384)  # n_items x text_dim
        >>> image_feat = torch.randn(7050, 4096)  # n_items x image_dim
        >>> fused = fusion(text_feat, image_feat)
        >>> print(fused.shape)  # torch.Size([7050, 128]) - concatenation of 64+64
    """
    
    def __init__(self, text_feat_dim, image_feat_dim, feat_embed_dim=64, output_dim=None):
        super(MultiModalFusion, self).__init__()
        
        self.text_feat_dim = text_feat_dim
        self.image_feat_dim = image_feat_dim
        self.feat_embed_dim = feat_embed_dim
        self.output_dim = output_dim
        
        # Text projection: text_feat_dim → feat_embed_dim
        if text_feat_dim is not None and text_feat_dim > 0:
            self.text_proj = nn.Linear(text_feat_dim, feat_embed_dim)
            nn.init.xavier_normal_(self.text_proj.weight)
        else:
            self.text_proj = None
        
        # Image projection: image_feat_dim → feat_embed_dim
        if image_feat_dim is not None and image_feat_dim > 0:
            self.image_proj = nn.Linear(image_feat_dim, feat_embed_dim)
            nn.init.xavier_normal_(self.image_proj.weight)
        else:
            self.image_proj = None
        
        # Calculate concatenated dimension
        concat_dim = 0
        if self.text_proj is not None:
            concat_dim += feat_embed_dim
        if self.image_proj is not None:
            concat_dim += feat_embed_dim
        
        # Optional projection after concatenation
        if output_dim is not None and output_dim != concat_dim:
            self.final_proj = nn.Linear(concat_dim, output_dim)
            nn.init.xavier_normal_(self.final_proj.weight)
        else:
            self.final_proj = None
        
        self.concat_dim = concat_dim
    
    def forward(self, text_feat=None, image_feat=None):
        """
        Fuse text and image features.
        
        Args:
            text_feat (torch.Tensor, optional): Text features of shape (n_items, text_feat_dim)
            image_feat (torch.Tensor, optional): Image features of shape (n_items, image_feat_dim)
        
        Returns:
            torch.Tensor: Fused features of shape (n_items, concat_dim) or (n_items, output_dim)
        
        Raises:
            ValueError: If both text_feat and image_feat are None
        """
        features = []
        
        # Project text features if available
        if text_feat is not None and self.text_proj is not None:
            text_embed = self.text_proj(text_feat)
            features.append(text_embed)
        
        # Project image features if available
        if image_feat is not None and self.image_proj is not None:
            image_embed = self.image_proj(image_feat)
            features.append(image_embed)
        
        if len(features) == 0:
            raise ValueError("At least one of text_feat or image_feat must be provided")
        
        # Concatenate features
        if len(features) == 1:
            fused = features[0]
        else:
            fused = torch.cat(features, dim=-1)
        
        # Optional final projection
        if self.final_proj is not None:
            fused = self.final_proj(fused)
        
        return fused
    
    def get_output_dim(self):
        """
        Get the output dimension of fused features.
        
        Returns:
            int: Output dimension
        """
        if self.output_dim is not None:
            return self.output_dim
        return self.concat_dim


# Test function for development/debugging
def test_simsiam_modules():
    """
    Test the SimSiam modules with sample data.
    """
    print("="*60)
    print("Testing SimSiam Modules")
    print("="*60)
    
    # Test SimSiamEncoder
    print("\n[1] Testing SimSiamEncoder")
    print("-" * 60)
    encoder = SimSiamEncoder(input_dim=128, hidden_dim=256, output_dim=128)
    x = torch.randn(32, 128)
    z = encoder(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {z.shape}")
    print(f"Expected:     torch.Size([32, 128])")
    print(f"✓ Shape matches!" if z.shape == torch.Size([32, 128]) else "✗ Shape mismatch!")
    
    # Test SimSiamPredictor
    print("\n[2] Testing SimSiamPredictor")
    print("-" * 60)
    predictor = SimSiamPredictor(input_dim=128, hidden_dim=64, output_dim=128)
    p = predictor(z)
    print(f"Input shape:  {z.shape}")
    print(f"Output shape: {p.shape}")
    print(f"Expected:     torch.Size([32, 128])")
    print(f"✓ Shape matches!" if p.shape == torch.Size([32, 128]) else "✗ Shape mismatch!")
    
    # Test MultiModalFusion with both modalities
    print("\n[3] Testing MultiModalFusion (text + image)")
    print("-" * 60)
    fusion = MultiModalFusion(text_feat_dim=384, image_feat_dim=4096, feat_embed_dim=64)
    text_feat = torch.randn(100, 384)
    image_feat = torch.randn(100, 4096)
    fused = fusion(text_feat, image_feat)
    print(f"Text input shape:  {text_feat.shape}")
    print(f"Image input shape: {image_feat.shape}")
    print(f"Fused shape:       {fused.shape}")
    print(f"Expected:          torch.Size([100, 128])")
    print(f"✓ Shape matches!" if fused.shape == torch.Size([100, 128]) else "✗ Shape mismatch!")
    
    # Test MultiModalFusion with text only
    print("\n[4] Testing MultiModalFusion (text only)")
    print("-" * 60)
    fusion_text = MultiModalFusion(text_feat_dim=384, image_feat_dim=None, feat_embed_dim=64)
    fused_text = fusion_text(text_feat=text_feat)
    print(f"Text input shape: {text_feat.shape}")
    print(f"Fused shape:      {fused_text.shape}")
    print(f"Expected:         torch.Size([100, 64])")
    print(f"✓ Shape matches!" if fused_text.shape == torch.Size([100, 64]) else "✗ Shape mismatch!")
    
    # Test MultiModalFusion with image only
    print("\n[5] Testing MultiModalFusion (image only)")
    print("-" * 60)
    fusion_image = MultiModalFusion(text_feat_dim=None, image_feat_dim=4096, feat_embed_dim=64)
    fused_image = fusion_image(image_feat=image_feat)
    print(f"Image input shape: {image_feat.shape}")
    print(f"Fused shape:       {fused_image.shape}")
    print(f"Expected:          torch.Size([100, 64])")
    print(f"✓ Shape matches!" if fused_image.shape == torch.Size([100, 64]) else "✗ Shape mismatch!")
    
    # Test gradient flow
    print("\n[6] Testing gradient flow through encoder → predictor")
    print("-" * 60)
    x = torch.randn(16, 128, requires_grad=True)
    z = encoder(x)
    p = predictor(z)
    loss = p.mean()
    loss.backward()
    print(f"Input requires_grad:  {x.requires_grad}")
    print(f"Encoder output grad:  {z.grad_fn is not None}")
    print(f"Predictor output grad: {p.grad_fn is not None}")
    print(f"Input has gradient:   {x.grad is not None}")
    print(f"✓ Gradients flow correctly!" if x.grad is not None else "✗ Gradient issue!")
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    test_simsiam_modules()
