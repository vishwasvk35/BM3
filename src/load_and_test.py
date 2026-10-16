#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Load and test a saved BM3 model checkpoint
"""

import torch
import argparse
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Load and test trained BM3 models

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.quick_start import quick_start
from utils.configurator import Config
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.utils import get_model, get_trainer
from utils.logger import init_logger
from logging import getLogger


def load_checkpoint(checkpoint_path):
    """
    Load a saved model checkpoint
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file
        
    Returns:
        Loaded checkpoint dictionary
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint (use CPU if GPU not available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Use weights_only=False for compatibility with older saved models
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    print(f"Checkpoint loaded successfully!")
    print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  - Best validation score: {checkpoint.get('best_valid_score', 'N/A')}")
    
    return checkpoint


def test_model(checkpoint_path, dataset_name='baby'):
    """
    Load a saved model and test it on the test set
    
    Args:
        checkpoint_path: Path to the saved model checkpoint
        dataset_name: Name of the dataset (baby, sports, or elec) - overrides checkpoint dataset
    """
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    
    # Get config from checkpoint
    config = checkpoint['config']
    
    # Override dataset name with the one we have
    original_dataset = config['dataset']
    config['dataset'] = dataset_name
    config['inter_file_name'] = f'{dataset_name}.inter'  # Override inter file name too
    
    # Make sure feature files are set for the new dataset
    config['vision_feature_file'] = 'image_feat.npy'
    config['text_feature_file'] = 'text_feat.npy'
    config['user_graph_dict_file'] = 'user_graph_dict.npy'
    
    print(f"  Note: Overriding dataset '{original_dataset}' -> '{dataset_name}'")
    
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nModel Configuration:")
    print(f"  - Model: {config['model']}")
    print(f"  - Dataset: {config['dataset']}")
    print(f"  - Embedding size: {config['embedding_size']}")
    try:
        print(f"  - n_layers: {config['n_layers']}")
        print(f"  - dropout: {config['dropout']}")
        print(f"  - reg_weight: {config['reg_weight']}")
    except (KeyError, AttributeError):
        print(f"  - Additional params: See config file")
    
    # Initialize logger
    init_logger(config)
    logger = getLogger()
    
    print("\nLoading dataset...")
    # Load dataset
    dataset = RecDataset(config)
    
    # Print dataset info (this initializes inter_num attribute)
    logger.info(str(dataset))
    
    # Split into train/val/test
    train_dataset, val_dataset, test_dataset = dataset.split()
    
    # Print split datasets (this initializes inter_num for each split)
    logger.info('\n====Training====\n' + str(train_dataset))
    logger.info('\n====Validation====\n' + str(val_dataset))
    logger.info('\n====Testing====\n' + str(test_dataset))
    
    # Create data loaders
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=False)
    test_data = EvalDataLoader(config, test_dataset, additional_dataset=train_dataset,
                               batch_size=config['eval_batch_size'], shuffle=False)
    
    print(f"\nDataset loaded: {config['dataset']}")
    print(f"  - Users: {train_dataset.user_num}")
    print(f"  - Items: {train_dataset.item_num}")
    print(f"  - Train interactions: {len(train_dataset)}")
    print(f"  - Test interactions: {len(test_dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = get_model(config['model'])(config, train_data).to(config['device'])
    
    # Load model state dict
    # Use strict=False to allow partial loading (in case of architecture differences)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    if missing_keys:
        print(f"\nWarning: Missing keys in checkpoint:")
        for key in missing_keys[:5]:  # Show first 5
            print(f"  - {key}")
        if len(missing_keys) > 5:
            print(f"  ... and {len(missing_keys)-5} more")
    
    if unexpected_keys:
        print(f"\nWarning: Unexpected keys in checkpoint (model was trained with additional features):")
        for key in unexpected_keys[:5]:  # Show first 5
            print(f"  - {key}")
        if len(unexpected_keys) > 5:
            print(f"  ... and {len(unexpected_keys)-5} more")
        print("\nNote: This usually means the trained model used different features.")
        print("      The model will still work but may not match the trained performance.")
    
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = get_trainer()(config, model)
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("Running evaluation on test set...")
    print("="*60)
    
    test_result = trainer.evaluate(test_data, is_test=True)
    
    # Print results
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    
    # Group by metric type
    metrics = ['recall', 'ndcg', 'precision', 'map']
    for metric in metrics:
        metric_results = {k: v for k, v in test_result.items() if metric in k.lower()}
        if metric_results:
            print(f"\n{metric.upper()}:")
            for k, v in sorted(metric_results.items()):
                print(f"  {k}: {v:.6f}")
    
    return test_result


def get_recommendations(checkpoint_path, user_ids, top_k=20, dataset_name='baby'):
    """
    Get top-K recommendations for specific users
    
    Args:
        checkpoint_path: Path to saved model checkpoint
        user_ids: List of user IDs to get recommendations for
        top_k: Number of recommendations to return
        dataset_name: Name of dataset
    
    Returns:
        Dictionary mapping user_id -> list of recommended item_ids with scores
    """
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint['config']
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load dataset
    dataset = RecDataset(config)
    train_data = TrainDataLoader(config, dataset, batch_size=2048, shuffle=False)
    
    # Load model
    model = get_model(config['model'])(config, train_data).to(config['device'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    print(f"\nGenerating recommendations for {len(user_ids)} users...")
    
    recommendations = {}
    
    with torch.no_grad():
        for user_id in user_ids:
            # Create interaction tensor
            interaction = torch.LongTensor([user_id]).to(config['device'])
            
            # Get prediction scores for all items
            scores = model.full_sort_predict(interaction)
            
            # Get top-K items
            top_scores, top_items = torch.topk(scores, top_k, dim=1)
            
            recommendations[user_id] = {
                'item_ids': top_items.cpu().numpy().tolist()[0],
                'scores': top_scores.cpu().numpy().tolist()[0]
            }
    
    return recommendations


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load and test BM3 model')
    parser.add_argument('-c', '--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('-d', '--dataset', type=str, default='baby',
                        choices=['baby', 'sports', 'elec'],
                        help='Dataset name')
    parser.add_argument('--mode', type=str, default='test',
                        choices=['test', 'recommend'],
                        help='Mode: test (evaluate on test set) or recommend (get recommendations)')
    parser.add_argument('--user_ids', type=int, nargs='+', default=[0, 1, 2, 3, 4],
                        help='User IDs for recommendation mode')
    parser.add_argument('--top_k', type=int, default=20,
                        help='Number of recommendations to generate')
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        # Test mode: evaluate on test set
        test_model(args.checkpoint, args.dataset)
    
    elif args.mode == 'recommend':
        # Recommendation mode: get recommendations for users
        recs = get_recommendations(args.checkpoint, args.user_ids, args.top_k, args.dataset)
        
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        
        for user_id, rec_data in recs.items():
            print(f"\nUser {user_id}:")
            print(f"  Top {args.top_k} recommended items:")
            for i, (item_id, score) in enumerate(zip(rec_data['item_ids'], rec_data['scores']), 1):
                print(f"    {i}. Item {item_id} (score: {score:.4f})")
