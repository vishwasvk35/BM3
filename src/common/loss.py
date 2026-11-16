# coding: utf-8
# @email  : enoche.chow@gmail.com


import torch
import torch.nn as nn


class BPRLoss(nn.Module):

    """ BPRLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """
    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = - torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss


class EmbLoss(nn.Module):
    """ EmbLoss, regularization on embeddings

    """
    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings):
        emb_loss = torch.zeros(1).to(embeddings[-1].device)
        for embedding in embeddings:
            emb_loss += torch.norm(embedding, p=self.norm)
        emb_loss /= embeddings[-1].shape[0]
        return emb_loss


class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, *embeddings):
        l2_loss = torch.zeros(1).to(embeddings[-1].device)
        for embedding in embeddings:
            l2_loss += torch.sum(embedding**2)*0.5
        return l2_loss


class SimSiamLoss(nn.Module):
    """ SimSiam Loss: Negative Cosine Similarity
    
    Implements the loss function used in SimSiam:
        L = -cosine_similarity(p, z.detach())
    
    where p is the predictor output and z is the encoder output (with stop-gradient).
    
    Args:
        - prediction: (N, D) predictor output
        - target: (N, D) encoder output (should be detached before passing)
        
    Returns:
        - loss: scalar, negative cosine similarity
        
    Example::
        >>> loss_fn = SimSiamLoss()
        >>> p = torch.randn(32, 128, requires_grad=True)
        >>> z = torch.randn(32, 128).detach()  # stop-gradient
        >>> loss = loss_fn(p, z)
        >>> loss.backward()
    """
    def __init__(self):
        super(SimSiamLoss, self).__init__()
    
    def forward(self, prediction, target):
        """
        Calculate negative cosine similarity.
        
        Args:
            prediction: (N, D) tensor, predictor output
            target: (N, D) tensor, encoder output (detached)
        
        Returns:
            torch.Tensor: scalar loss
        """
        # Negative cosine similarity
        loss = -torch.nn.functional.cosine_similarity(prediction, target, dim=-1).mean()
        return loss