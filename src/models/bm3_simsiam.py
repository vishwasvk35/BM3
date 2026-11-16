# coding: utf-8
# BM3 with SimSiam Integration
# Combines Bootstrap Multi-Modal Multi-task learning with SimSiam's stop-gradient mechanism

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity

from common.abstract_recommender import GeneralRecommender
from common.loss import EmbLoss
from models.simsiam_modules import SimSiamEncoder, SimSiamPredictor, MultiModalFusion


class BM3SimSiam(GeneralRecommender):
    """
    BM3 with SimSiam Integration for Multi-Modal Recommendation.
    
    This model enhances BM3 by:
    1. Replacing momentum-based target network with SimSiam's stop-gradient mechanism
    2. Using SimSiam's encoder-predictor architecture for feature refinement
    3. Fusing text and image features before applying transformations
    4. Joint training with recommendation loss and SimSiam contrastive loss
    
    Architecture:
        - User/Item ID embeddings → LightGCN propagation
        - Text + Image features → MultiModalFusion → Concatenate with ID embeddings
        - SimSiam encoder → stop-gradient target
        - SimSiam predictor → predictions
        - Loss: recommendation + SimSiam contrastive + regularization
    
    Args:
        config: Configuration dictionary containing hyperparameters
        dataset: Dataset object with user-item interactions and multi-modal features
    """
    
    def __init__(self, config, dataset):
        super(BM3SimSiam, self).__init__(config, dataset)
        
        # Basic parameters
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim'] if 'feat_embed_dim' in config else 64
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.dropout = config['dropout'] if 'dropout' in config else 0.1
        
        # SimSiam specific parameters
        self.simsiam_weight = config['simsiam_weight'] if 'simsiam_weight' in config else 1.0
        self.simsiam_hidden_dim = config['simsiam_hidden_dim'] if 'simsiam_hidden_dim' in config else 64
        self.simsiam_output_dim = config['simsiam_output_dim'] if 'simsiam_output_dim' in config else 128
        
        self.n_nodes = self.n_users + self.n_items
        
        # Load normalized adjacency matrix for LightGCN
        self.norm_adj = self.get_norm_adj_mat(
            dataset.inter_matrix(form='coo').astype(np.float32)
        ).to(self.device)
        
        # ========== Embedding Layers ==========
        # User and Item ID embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        
        # ========== Multi-Modal Feature Processing ==========
        # Determine feature dimensions
        text_feat_dim = self.t_feat.shape[1] if self.t_feat is not None else None
        image_feat_dim = self.v_feat.shape[1] if self.v_feat is not None else None
        
        # Initialize feature embeddings
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
        else:
            self.text_embedding = None
        
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
        else:
            self.image_embedding = None
        
        # Multi-modal fusion layer
        if text_feat_dim is not None or image_feat_dim is not None:
            self.fusion = MultiModalFusion(
                text_feat_dim=text_feat_dim,
                image_feat_dim=image_feat_dim,
                feat_embed_dim=self.feat_embed_dim,
                output_dim=None  # Output will be concatenation of projections
            )
            self.fused_feat_dim = self.fusion.get_output_dim()
        else:
            self.fusion = None
            self.fused_feat_dim = 0
        
        # ========== SimSiam Components ==========
        # Determine input dimension for SimSiam
        # For items: ID embedding + fused multi-modal features
        # For users: ID embedding only
        self.item_total_dim = self.embedding_dim + self.fused_feat_dim
        self.user_total_dim = self.embedding_dim
        
        # SimSiam Encoder: projects embeddings to representation space
        self.encoder = SimSiamEncoder(
            input_dim=self.item_total_dim,
            hidden_dim=self.simsiam_output_dim * 2,  # 256 for 128 output
            output_dim=self.simsiam_output_dim
        )
        
        # SimSiam Predictor: asymmetric prediction head
        self.predictor = SimSiamPredictor(
            input_dim=self.simsiam_output_dim,
            hidden_dim=self.simsiam_hidden_dim,
            output_dim=self.simsiam_output_dim
        )
        
        # User predictor (simpler, just for recommendation task)
        self.user_predictor = nn.Linear(self.user_total_dim, self.simsiam_output_dim)
        nn.init.xavier_normal_(self.user_predictor.weight)
        
        # ========== Loss Functions ==========
        self.reg_loss = EmbLoss()
    
    def get_norm_adj_mat(self, interaction_matrix):
        """
        Get normalized adjacency matrix for LightGCN.
        
        Creates a symmetric normalized adjacency matrix:
            L = D^(-1/2) * A * D^(-1/2)
        
        where A is the user-item bipartite graph adjacency matrix.
        
        Args:
            interaction_matrix: Sparse user-item interaction matrix (COO format)
        
        Returns:
            torch.sparse.FloatTensor: Normalized adjacency matrix
        """
        # Create bipartite graph adjacency matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items),
            dtype=np.float32
        )
        
        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()
        
        # Add edges: users to items and items to users
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz))
        )
        A.update(data_dict)
        
        # Normalize: D^(-1/2) * A * D^(-1/2)
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7  # Add epsilon to avoid division by zero
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        
        # Convert to PyTorch sparse tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        
        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))
    
    def get_fused_item_features(self):
        """
        Get fused multi-modal features for all items.
        
        Returns:
            torch.Tensor: Fused features of shape (n_items, fused_feat_dim)
                         Returns None if no multi-modal features available
        """
        if self.fusion is None:
            return None
        
        text_feat = self.text_embedding.weight if self.text_embedding is not None else None
        image_feat = self.image_embedding.weight if self.image_embedding is not None else None
        
        if text_feat is None and image_feat is None:
            return None
        
        fused_feat = self.fusion(text_feat, image_feat)
        return fused_feat
    
    def forward(self):
        """
        Forward propagation through the model.
        
        Steps:
        1. Fuse multi-modal features (text + image)
        2. Combine user ID embeddings and item (ID + fused features)
        3. Propagate through LightGCN layers
        4. Split back into user and item embeddings
        5. Add residual connection for items
        
        Returns:
            tuple: (user_embeddings, item_embeddings)
                - user_embeddings: (n_users, embedding_dim)
                - item_embeddings: (n_items, item_total_dim)
        """
        # Get fused multi-modal features for items
        fused_item_feat = self.get_fused_item_features()
        
        # Prepare item features: ID embedding + fused multi-modal features
        if fused_item_feat is not None:
            # Concatenate ID embeddings with fused features
            item_features = torch.cat([self.item_id_embedding.weight, fused_item_feat], dim=-1)
        else:
            item_features = self.item_id_embedding.weight
        
        # Combine user and item embeddings for graph propagation
        # Note: For graph propagation, we need same dimensions for users and items
        # So we'll propagate ID embeddings only through the graph
        ego_embeddings = torch.cat(
            (self.user_embedding.weight, self.item_id_embedding.weight), dim=0
        )
        
        # LightGCN: Graph convolution
        all_embeddings = [ego_embeddings]
        for i in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        
        # Mean pooling across layers
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        
        # Split back into user and item embeddings
        u_g_embeddings, i_g_embeddings = torch.split(
            all_embeddings, [self.n_users, self.n_items], dim=0
        )
        
        # For items: combine graph-propagated ID embedding with fused features
        if fused_item_feat is not None:
            # Add residual connection: use graph-enhanced ID embedding + original fused features
            item_embeddings = torch.cat([i_g_embeddings, fused_item_feat], dim=-1)
        else:
            item_embeddings = i_g_embeddings
        
        return u_g_embeddings, item_embeddings
    
    def calculate_simsiam_loss(self, embeddings):
        """
        Calculate SimSiam contrastive loss using stop-gradient.
        
        SimSiam loss: D(p, stop_gradient(z))
        where:
            - z = encoder(embedding)
            - p = predictor(z)
            - D = negative cosine similarity
        
        Args:
            embeddings: Input embeddings (item embeddings)
        
        Returns:
            torch.Tensor: SimSiam contrastive loss (scalar)
        """
        # Encode: embeddings → z
        z = self.encoder(embeddings)
        
        # Predict: z → p
        p = self.predictor(z)
        
        # Stop gradient on z to create target
        z_target = z.detach()
        
        # Negative cosine similarity
        loss = -(F.cosine_similarity(p, z_target, dim=-1).mean())
        
        return loss
    
    def calculate_recommendation_loss(self, user_embeddings, item_embeddings, interactions):
        """
        Calculate recommendation loss (user-item alignment).
        
        Args:
            user_embeddings: User embeddings from forward pass
            item_embeddings: Item embeddings from forward pass
            interactions: Batch of user-item interactions
        
        Returns:
            torch.Tensor: Recommendation loss (scalar)
        """
        users, items = interactions[0], interactions[1]
        
        # Get embeddings for current batch
        batch_user_emb = user_embeddings[users, :]
        batch_item_emb = item_embeddings[items, :]
        
        # Project to prediction space
        user_pred = self.user_predictor(batch_user_emb)
        
        # For items, use encoder → predictor
        item_encoded = self.encoder(batch_item_emb)
        item_pred = self.predictor(item_encoded)
        
        # Cosine similarity between user and item predictions
        similarity = F.cosine_similarity(user_pred, item_pred, dim=-1)
        
        # Loss: 1 - similarity (we want to maximize similarity)
        loss = (1 - similarity).mean()
        
        return loss
    
    def calculate_loss(self, interactions):
        """
        Calculate total loss combining recommendation, SimSiam, and regularization.
        
        Total Loss = L_recommendation + λ_simsiam * L_simsiam + λ_reg * L_reg
        
        Args:
            interactions: Batch of user-item interactions
        
        Returns:
            torch.Tensor: Total loss (scalar)
        """
        # Forward pass
        user_embeddings, item_embeddings = self.forward()
        
        # 1. Recommendation loss (user-item alignment)
        loss_rec = self.calculate_recommendation_loss(
            user_embeddings, item_embeddings, interactions
        )
        
        # 2. SimSiam contrastive loss on item embeddings
        # Sample items from the batch to reduce computation
        items = interactions[1]
        batch_item_emb = item_embeddings[items, :]
        loss_simsiam = self.calculate_simsiam_loss(batch_item_emb)
        
        # 3. Regularization loss (L2 on base embeddings)
        loss_reg = self.reg_loss(
            self.user_embedding.weight[interactions[0]],
            self.item_id_embedding.weight[interactions[1]]
        )
        
        # Total loss
        total_loss = (
            loss_rec +
            self.simsiam_weight * loss_simsiam +
            self.reg_weight * loss_reg
        )
        
        return total_loss
    
    def full_sort_predict(self, interaction):
        """
        Predict scores for all items given user(s).
        
        Used during evaluation to rank all items for each user.
        
        Args:
            interaction: User interaction(s) to predict for
        
        Returns:
            torch.Tensor: Score matrix (batch_size x n_items)
        """
        user = interaction[0]
        
        # Forward pass
        user_embeddings, item_embeddings = self.forward()
        
        # Project to prediction space
        user_pred = self.user_predictor(user_embeddings[user])
        
        # For items, encode and predict
        item_encoded = self.encoder(item_embeddings)
        item_pred = self.predictor(item_encoded)
        
        # Compute scores: dot product between user and item predictions
        score_mat = torch.matmul(user_pred, item_pred.transpose(0, 1))
        
        return score_mat
