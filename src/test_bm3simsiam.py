# coding: utf-8
# Test script for BM3SimSiam model

import sys
import os
import torch
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from utils.configurator import Config
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader
from utils.utils import init_seed, get_model

def test_model_instantiation():
    """Test BM3SimSiam model instantiation and basic operations."""
    
    print("="*80)
    print("BM3SimSiam Model Test")
    print("="*80)
    
    # Configuration
    config_dict = {
        'gpu_id': -1,  # Use CPU for testing
    }
    
    model_name = 'BM3SimSiam'
    dataset_name = 'baby'
    
    # Load configuration
    print("\n[1] Loading configuration...")
    config = Config(model_name, dataset_name, config_dict)
    init_seed(config['seed'])
    print(f"✓ Configuration loaded")
    print(f"  - Model: {model_name}")
    print(f"  - Dataset: {dataset_name}")
    print(f"  - Device: {config['device']}")
    
    # Load dataset
    print("\n[2] Loading dataset...")
    dataset = RecDataset(config)
    print(f"✓ Dataset loaded")
    print(f"  - Users: {dataset.get_user_num()}")
    print(f"  - Items: {dataset.get_item_num()}")
    print(f"  - Interactions: {dataset.inter_num()}")
    
    # Split dataset
    train_dataset, valid_dataset, test_dataset = dataset.split()
    print(f"  - Train: {train_dataset.inter_num()} interactions")
    print(f"  - Valid: {valid_dataset.inter_num()} interactions")
    print(f"  - Test: {test_dataset.inter_num()} interactions")
    
    # Create dataloader
    print("\n[3] Creating dataloader...")
    train_data = TrainDataLoader(
        config, train_dataset, 
        batch_size=config['train_batch_size'], 
        shuffle=True
    )
    train_data.pretrain_setup()
    print(f"✓ Dataloader created")
    print(f"  - Batch size: {config['train_batch_size']}")
    print(f"  - Number of batches: {len(train_data)}")
    
    # Instantiate model
    print("\n[4] Instantiating BM3SimSiam model...")
    model_class = get_model(model_name)
    model = model_class(config, train_data).to(config['device'])
    print(f"✓ Model instantiated successfully")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
    # Print model architecture
    print("\n[5] Model Architecture:")
    print("-" * 80)
    print(model)
    print("-" * 80)
    
    # Test forward pass
    print("\n[6] Testing forward pass...")
    try:
        user_embeddings, item_embeddings = model.forward()
        print(f"✓ Forward pass successful")
        print(f"  - User embeddings shape: {user_embeddings.shape}")
        print(f"  - Item embeddings shape: {item_embeddings.shape}")
        print(f"  - User embedding stats: mean={user_embeddings.mean().item():.4f}, std={user_embeddings.std().item():.4f}")
        print(f"  - Item embedding stats: mean={item_embeddings.mean().item():.4f}, std={item_embeddings.std().item():.4f}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test loss calculation
    print("\n[7] Testing loss calculation...")
    try:
        # Get a batch
        for batch_idx, interaction in enumerate(train_data):
            if batch_idx == 0:  # Only test first batch
                loss = model.calculate_loss(interaction)
                print(f"✓ Loss calculation successful")
                print(f"  - Loss value: {loss.item():.4f}")
                print(f"  - Loss requires_grad: {loss.requires_grad}")
                break
    except Exception as e:
        print(f"✗ Loss calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test backward pass
    print("\n[8] Testing backward pass...")
    try:
        loss.backward()
        print(f"✓ Backward pass successful")
        
        # Check if gradients are computed
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad = True
                break
        
        if has_grad:
            print(f"  - Gradients computed successfully")
        else:
            print(f"  - Warning: No gradients found")
    except Exception as e:
        print(f"✗ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test prediction
    print("\n[9] Testing full_sort_predict...")
    try:
        # Get first user
        test_interaction = [torch.tensor([0]).to(config['device'])]
        scores = model.full_sort_predict(test_interaction)
        print(f"✓ Prediction successful")
        print(f"  - Scores shape: {scores.shape}")
        print(f"  - Score range: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
        
        # Get top-10 items
        top_k = 10
        top_scores, top_items = torch.topk(scores[0], top_k)
        print(f"  - Top-{top_k} items for user 0:")
        for i in range(min(5, top_k)):
            print(f"    - Item {top_items[i].item()}: score {top_scores[i].item():.4f}")
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*80)
    print("All tests passed! ✓")
    print("="*80)


if __name__ == "__main__":
    test_model_instantiation()
