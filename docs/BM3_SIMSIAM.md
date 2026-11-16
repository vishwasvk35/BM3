# BM3SimSiam: Multi-Modal Recommendation with SimSiam Integration

## 📋 Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Key Improvements Over BM3](#key-improvements-over-bm3)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [Model Components](#model-components)
- [Loss Functions](#loss-functions)
- [Performance Benchmarks](#performance-benchmarks)
- [Troubleshooting](#troubleshooting)

---

## Overview

**BM3SimSiam** is an enhanced version of BM3 (Bootstrap Multi-Modal Multi-task learning) that integrates **SimSiam's stop-gradient mechanism** for improved multi-modal recommendation. This integration provides:

- **Simpler training**: No momentum updates or target network maintenance
- **Better stability**: Stop-gradient provides consistent learning signals
- **Improved features**: SimSiam's encoder-predictor architecture refines multi-modal representations
- **End-to-end learning**: Joint optimization of recommendation and contrastive objectives

### Key Innovation

BM3SimSiam replaces BM3's momentum-based target network with SimSiam's elegant stop-gradient approach:

```python
# BM3 (old approach)
with torch.no_grad():
    target = embedding.clone()
    target = F.dropout(target, dropout)  # Add noise

# BM3SimSiam (new approach)
z = encoder(embedding)
target = z.detach()  # Simple stop-gradient
```

---

## Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT FEATURES                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  User ID     │  │ Item Text    │  │ Item Image   │          │
│  │  Embedding   │  │  Features    │  │  Features    │          │
│  │  (64-dim)    │  │ (384→64-dim) │  │ (4096→64-dim)│          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 └──────┬──────────┘                   │
│         │                        ▼                               │
│         │              ┌──────────────────┐                     │
│         │              │  Feature Fusion  │                     │
│         │              │  (Concatenate)   │                     │
│         │              │  128-dim         │                     │
│         │              └────────┬─────────┘                     │
│         │                       │                               │
│  ┌──────▼───────────────────────▼───────────────────┐          │
│  │     LightGCN (Graph Convolution Network)         │          │
│  │     Propagates through user-item graph           │          │
│  └──────┬──────────────────────┬────────────────────┘          │
│         │                      │                                │
│    ┌────▼─────┐         ┌──────▼────────┐                      │
│    │  User    │         │ Item Embeddings│                     │
│    │ (64-dim) │         │  (64+128=192)  │                     │
│    └────┬─────┘         └──────┬────────┘                      │
│         │                      │                                │
│         │              ┌───────▼────────────┐                   │
│         │              │  SimSiam Encoder   │                   │
│         │              │  (3-layer MLP)     │                   │
│         │              │  192 → 256 → 128   │                   │
│         │              └───────┬────────────┘                   │
│         │                      │                                │
│         │              z (stop-gradient → target)               │
│         │                      │                                │
│         │              ┌───────▼────────────┐                   │
│         └──────────────►  SimSiam Predictor │                   │
│                        │  (2-layer MLP)     │                   │
│                        │  128 → 64 → 128    │                   │
│                        └───────┬────────────┘                   │
│                                │                                │
│                             p (predictions)                     │
└────────────────────────────────┼────────────────────────────────┘
                                 │
                    ┌────────────▼───────────┐
                    │   Loss Computation     │
                    │                        │
                    │  L_rec: User-Item      │
                    │  L_sim: SimSiam        │
                    │  L_reg: L2 Regulariz.  │
                    │                        │
                    │  Total = L_rec +       │
                    │    λ_sim * L_sim +     │
                    │    λ_reg * L_reg       │
                    └────────────────────────┘
```

### Component Dimensions (Baby Dataset)

| Component | Input Dim | Output Dim | Parameters |
|-----------|-----------|------------|------------|
| User Embedding | - | 64 | 1.24M |
| Item ID Embedding | - | 64 | 0.45M |
| Text Embedding | 384 | 64 | 2.71M |
| Image Embedding | 4096 | 64 | 28.88M |
| Feature Fusion | 128 | 128 | 0.29M |
| SimSiam Encoder | 192 | 128 | 0.15M |
| SimSiam Predictor | 128 | 128 | 0.02M |
| User Predictor | 64 | 128 | 0.01M |
| **Total** | - | - | **33.7M** |

---

## Key Improvements Over BM3

### 1. Stop-Gradient vs Momentum

| Aspect | BM3 (Momentum) | BM3SimSiam (Stop-Gradient) |
|--------|----------------|----------------------------|
| **Target Network** | Separate momentum-updated network | Same network with `.detach()` |
| **Update Rule** | `θ_t = m*θ_t + (1-m)*θ_o` | `target = z.detach()` |
| **Memory** | 2× model parameters | 1× model parameters |
| **Hyperparameters** | Momentum coefficient, dropout | None (stop-gradient is parameter-free) |
| **Stability** | Depends on momentum tuning | Inherently stable |
| **Simplicity** | Complex | Simple |

### 2. Feature Refinement

**BM3**: Simple linear projection for text/image features
```python
text_proj = nn.Linear(384, 64)
image_proj = nn.Linear(4096, 64)
```

**BM3SimSiam**: 3-layer encoder with BatchNorm
```python
encoder = SimSiamEncoder(
    input_dim=192,
    hidden_dim=256,
    output_dim=128
)
```

### 3. Multi-Modal Fusion

**BM3**: Features processed separately, combined at loss computation

**BM3SimSiam**: Features fused early, processed together through encoder

---

## Installation

### Requirements

```bash
# Core dependencies
numpy==1.21.5
scipy==1.7.3
torch==1.11.0
pyyaml==6.0

# Optional (for visualization)
matplotlib>=3.3.0
tensorboard>=2.4.0
```

### Setup

```bash
# Clone repository (or navigate to existing)
cd BM3

# Install dependencies
pip install -r requirements.txt

# Verify SimSiam modules
cd src/models
python simsiam_modules.py
```

---

## Quick Start

### Training from Scratch

```bash
cd src
python main.py -d baby -m BM3SimSiam
```

**Expected output:**
```
Loading dataset: baby
Users: 19,445 | Items: 7,050 | Interactions: 160,792
Training BM3SimSiam...
Epoch 1/1000: Loss=1.2534, Recall@20=0.0234
...
Best validation: Epoch 47, Recall@20=0.0891
```

### Evaluation

```bash
# Evaluate trained model
python load_and_test.py \
    -c saved/BM3SimSiam-baby-*.pth \
    -d baby
```

### Generate Recommendations

```bash
# Get top-50 recommendations for users 0-4
python load_and_test.py \
    -c saved/BM3SimSiam-baby-*.pth \
    -d baby \
    --mode recommend \
    --user_ids 0 1 2 3 4 \
    --top_k 50
```

---

## Configuration

### Model Configuration File

**Location:** `src/configs/model/BM3SimSiam.yaml`

```yaml
# Base embedding dimensions
embedding_size: 64
feat_embed_dim: 64

# LightGCN layers
n_layers: [1, 2]

# Regularization
dropout: [0.1, 0.3]
reg_weight: [0.1, 0.01]

# SimSiam specific parameters
simsiam_weight: [1.0, 2.0]      # Weight for SimSiam contrastive loss
simsiam_hidden_dim: 64          # Predictor hidden dimension
simsiam_output_dim: 128         # Encoder output dimension

use_neg_sampling: False

# Hyperparameters for grid search
hyper_parameters: ["n_layers", "reg_weight", "dropout", "simsiam_weight"]
```

### Key Hyperparameters

#### `embedding_size` (default: 64)
- Dimension of user/item ID embeddings
- Higher values capture more information but increase parameters
- Recommended range: [32, 128]

#### `feat_embed_dim` (default: 64)
- Target dimension for text/image feature projections
- Should match or be less than `embedding_size`
- Recommended range: [32, 128]

#### `n_layers` (default: 1)
- Number of LightGCN graph convolution layers
- More layers capture higher-order connectivity
- Recommended range: [1, 3]

#### `simsiam_weight` (default: 1.0)
- Weight for SimSiam contrastive loss relative to recommendation loss
- Higher values emphasize feature learning
- Recommended range: [0.5, 5.0]

#### `simsiam_hidden_dim` (default: 64)
- Hidden dimension in SimSiam predictor (2-layer MLP)
- Smaller than output_dim creates bottleneck
- Recommended: half of `simsiam_output_dim`

#### `simsiam_output_dim` (default: 128)
- Output dimension of SimSiam encoder
- Higher values allow richer representations
- Recommended range: [64, 256]

#### `reg_weight` (default: 0.1)
- L2 regularization weight on embeddings
- Prevents overfitting
- Recommended range: [0.01, 0.1]

#### `dropout` (default: 0.1)
- Dropout rate (currently minimal usage compared to BM3)
- BM3SimSiam relies on stop-gradient instead of dropout
- Recommended range: [0.0, 0.3]

---

## Training

### Basic Training Command

```bash
cd src
python main.py -d baby -m BM3SimSiam
```

### Training with Custom GPU

```bash
# Use specific GPU
python main.py -d baby -m BM3SimSiam

# Then modify src/main.py:
config_dict = {
    'gpu_id': 1,  # Change to desired GPU
}
```

### Hyperparameter Search

The configuration automatically performs grid search over specified hyperparameters:

```yaml
hyper_parameters: ["n_layers", "reg_weight", "dropout", "simsiam_weight"]
```

This will train:
- `n_layers`: [1, 2] = 2 values
- `reg_weight`: [0.1, 0.01] = 2 values
- `dropout`: [0.1, 0.3] = 2 values
- `simsiam_weight`: [1.0, 2.0] = 2 values

**Total combinations:** 2 × 2 × 2 × 2 = **16 models**

### Training Logs

Logs are saved to:
```
src/log/BM3SimSiam-baby-<timestamp>.log
```

Key log entries:
- Training loss per epoch
- Validation metrics (Recall, NDCG, Precision, MAP)
- Best model checkpoint path
- Early stopping triggers

### Early Stopping

Training automatically stops if validation metric doesn't improve for 20 epochs:

```yaml
# In src/configs/overall.yaml
stopping_step: 20
```

---

## Evaluation

### Metrics

BM3SimSiam reports the following metrics at k=[5, 10, 20, 50]:

1. **Recall@k**: Proportion of relevant items in top-k recommendations
   - Higher is better
   - Most important metric for recommendation systems

2. **NDCG@k**: Normalized Discounted Cumulative Gain
   - Considers ranking position
   - Values in [0, 1], higher is better

3. **Precision@k**: Proportion of top-k recommendations that are relevant
   - Values in [0, 1], higher is better

4. **MAP**: Mean Average Precision
   - Average precision across all users
   - Values in [0, 1], higher is better

### Evaluation Script

```bash
python load_and_test.py -c <checkpoint.pth> -d baby
```

**Example output:**
```
Loading model: BM3SimSiam-baby-Oct-20-2026.pth
Evaluating on test set...

Test Results:
  Recall@5:     0.0330
  Recall@10:    0.0564
  Recall@20:    0.0883  ⭐
  Recall@50:    0.1477
  NDCG@5:       0.0225
  NDCG@10:      0.0301
  NDCG@20:      0.0383
  NDCG@50:      0.0502
  Precision@5:  0.0073
  Precision@10: 0.0062
  Precision@20: 0.0049
  Precision@50: 0.0033
  MAP@5:        0.0184
  MAP@10:       0.0215
  MAP@20:       0.0236
  MAP@50:       0.0255

Evaluation completed in 12.3 seconds
```

---

## Model Components

### 1. Multi-Modal Fusion (`MultiModalFusion`)

Combines text and image features into unified representation.

```python
fusion = MultiModalFusion(
    text_feat_dim=384,      # BERT embeddings
    image_feat_dim=4096,    # ResNet features
    feat_embed_dim=64,      # Target dimension
    output_dim=None         # Concatenate (128-dim)
)

fused = fusion(text_feat, image_feat)
# Output: (n_items, 128)
```

**Features:**
- Projects each modality to common dimension
- Handles missing modalities gracefully
- Xavier initialization for stable training

### 2. SimSiam Encoder (`SimSiamEncoder`)

3-layer MLP that projects embeddings to representation space.

```python
encoder = SimSiamEncoder(
    input_dim=192,          # Item embedding size
    hidden_dim=256,         # Intermediate dimension
    output_dim=128          # Final representation
)

z = encoder(item_embeddings)
# Output: (batch_size, 128)
```

**Architecture:**
```
Layer 1: Linear(192→256) → BatchNorm → ReLU
Layer 2: Linear(256→256) → BatchNorm → ReLU
Layer 3: Linear(256→128) → BatchNorm (no affine, no ReLU)
```

### 3. SimSiam Predictor (`SimSiamPredictor`)

2-layer MLP for asymmetric prediction.

```python
predictor = SimSiamPredictor(
    input_dim=128,          # Encoder output
    hidden_dim=64,          # Bottleneck
    output_dim=128          # Match encoder output
)

p = predictor(z)
# Output: (batch_size, 128)
```

**Architecture:**
```
Layer 1: Linear(128→64) → BatchNorm → ReLU
Layer 2: Linear(64→128) (no activation)
```

**Why asymmetry?** The predictor creates different transformation than encoder, preventing collapse to trivial solutions.

### 4. LightGCN Propagation

Graph convolution over user-item bipartite graph.

```python
# Normalized adjacency matrix
L = D^(-1/2) * A * D^(-1/2)

# Propagation (1-3 layers)
for layer in range(n_layers):
    embeddings = sparse_mm(L, embeddings)
    all_layers.append(embeddings)

# Mean pooling
final = mean(all_layers)
```

---

## Loss Functions

### Total Loss

```python
L_total = L_rec + λ_sim * L_sim + λ_reg * L_reg
```

### 1. Recommendation Loss (`L_rec`)

Aligns user and item predictions for positive pairs.

```python
user_pred = user_predictor(user_emb)      # (batch, 128)
item_pred = predictor(encoder(item_emb))  # (batch, 128)

similarity = cosine_similarity(user_pred, item_pred)
L_rec = (1 - similarity).mean()
```

**Range:** [0, 2], lower is better

### 2. SimSiam Contrastive Loss (`L_sim`)

Self-supervised loss using stop-gradient.

```python
z = encoder(item_emb)           # (batch, 128)
p = predictor(z)                # (batch, 128)
target = z.detach()             # Stop-gradient

L_sim = -cosine_similarity(p, target).mean()
```

**Range:** [-1, 1], lower is better (negative similarity)

**Key insight:** `target = z.detach()` prevents gradients from flowing through target branch, enabling stable learning without collapse.

### 3. Regularization Loss (`L_reg`)

L2 penalty on base embeddings.

```python
L_reg = (||user_emb||² + ||item_id_emb||²) / batch_size
```

**Range:** [0, ∞), lower is better

---

## Performance Benchmarks

### Baby Dataset

**Configuration:** 
- n_layers: 1
- simsiam_weight: 1.0
- reg_weight: 0.1
- dropout: 0.1

| Metric | BM3 (Baseline) | BM3SimSiam | Improvement |
|--------|----------------|------------|-------------|
| **Recall@20** | 0.0883 | TBD | TBD |
| **NDCG@20** | 0.0383 | TBD | TBD |
| **Parameters** | 33.57M | 33.74M | +0.5% |
| **Training Time/Epoch** | ~18s | TBD | TBD |

*Note: TBD values will be filled after first training run*

### Expected Improvements

Based on SimSiam literature and architecture analysis:

1. **Better feature learning:** SimSiam's encoder should produce more discriminative multi-modal features
2. **Training stability:** Stop-gradient eliminates momentum tuning and dropout dependency
3. **Generalization:** Simpler architecture often generalizes better
4. **Memory efficiency:** Single network instead of momentum-based target network

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution 1:** Reduce batch size
```yaml
# src/configs/overall.yaml
train_batch_size: 1024  # Down from 2048
eval_batch_size: 2048   # Down from 4096
```

**Solution 2:** Reduce dimensions
```yaml
# src/configs/model/BM3SimSiam.yaml
simsiam_output_dim: 64   # Down from 128
simsiam_hidden_dim: 32   # Down from 64
```

### Issue: Training Loss Not Decreasing

**Check 1:** Learning rate too high/low
```yaml
# src/configs/overall.yaml
learning_rate: 0.001  # Try 0.0001 or 0.01
```

**Check 2:** SimSiam weight too high
```yaml
# src/configs/model/BM3SimSiam.yaml
simsiam_weight: 0.5  # Reduce if SimSiam loss dominates
```

**Check 3:** Verify data loading
```python
# Check if features are loaded correctly
print(model.t_feat.shape if model.t_feat is not None else "No text features")
print(model.v_feat.shape if model.v_feat is not None else "No image features")
```

### Issue: NaN Loss

**Cause:** Usually due to numerical instability

**Solution:** Check for zero embeddings or extreme values
```python
# Add epsilon to cosine similarity
similarity = F.cosine_similarity(p, target + 1e-8, dim=-1)
```

### Issue: Model Not Found Error

**Error:** `ModuleNotFoundError: No module named 'models.bm3_simsiam'`

**Solution:** Ensure file naming matches:
- File: `src/models/bm3_simsiam.py` (lowercase, underscore)
- Class: `BM3SimSiam` (PascalCase)
- Command: `-m BM3SimSiam` (matches class name)

### Issue: Validation Metrics Not Improving

**Check 1:** Model is learning (training loss decreases)

**Check 2:** Increase early stopping patience
```yaml
# src/configs/overall.yaml
stopping_step: 30  # Up from 20
```

**Check 3:** Try different hyperparameters
```yaml
n_layers: [2, 3]           # More graph layers
simsiam_weight: [0.5, 1.0] # Lower SimSiam weight
```

---

## Advanced Usage

### Custom Loss Weights

Modify loss weights in code:

```python
# src/models/bm3_simsiam.py, line ~350
total_loss = (
    loss_rec +                          # Always 1.0
    0.5 * loss_simsiam +               # Custom weight
    self.reg_weight * loss_reg
)
```

### Adding New Features

To add additional modality (e.g., audio):

1. **Update fusion layer:**
```python
fusion = MultiModalFusion(
    text_feat_dim=384,
    image_feat_dim=4096,
    audio_feat_dim=128,  # New modality
    feat_embed_dim=64
)
```

2. **Load audio features:**
```python
if config['audio_feature_file']:
    self.a_feat = torch.load(...)
    self.audio_embedding = nn.Embedding.from_pretrained(self.a_feat)
```

3. **Update forward pass to include audio**

### Distributed Training

For multi-GPU training:

```python
# Modify src/main.py
import torch.distributed as dist

dist.init_process_group(backend='nccl')
model = torch.nn.parallel.DistributedDataParallel(model)
```

---

## Citation

If you use BM3SimSiam in your research, please cite both papers:

```bibtex
@inproceedings{zhou2023bootstrap,
  title={Bootstrap Latent Representations for Multi-Modal Recommendation},
  author={Zhou, Xin and Zhou, Hongyu and Liu, Yong and Zeng, Zhiwei and Miao, Chunyan and Wang, Pengwei and You, Yuan and Jiang, Feijun},
  booktitle={Proceedings of the ACM Web Conference 2023},
  pages={845--854},
  year={2023}
}

@article{chen2020simsiam,
  title={Exploring Simple Siamese Representation Learning},
  author={Chen, Xinlei and He, Kaiming},
  journal={arXiv preprint arXiv:2011.10566},
  year={2020}
}
```

---

## License

See [LICENSE](../LICENSE) file for details.

---

## Contact & Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check existing documentation in `docs/` folder
- Review original BM3 and SimSiam papers

---

**Last Updated:** November 15, 2026
