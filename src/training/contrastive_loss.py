"""
contrastive_loss.py
Subject-aware contrastive regularization for cross-subject generalization.

The key idea: push embeddings of the SAME emotion class from DIFFERENT subjects
closer together, while pushing DIFFERENT emotion classes apart. This forces the
network to learn subject-invariant emotional representations.

Usage in training loop:
    from src.training.contrastive_loss import SubjectContrastiveLoss

    contrastive_fn = SubjectContrastiveLoss(temperature=0.1)

    # In training step:
    embedding = model.extract_embedding(eeg, ecg)  # (batch, embed_dim)
    ce_loss = criterion(logits, labels)
    cl_loss = contrastive_fn(embedding, labels, subjects)
    loss = ce_loss + 0.3 * cl_loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SubjectContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss that is subject-aware.

    For each anchor sample, positives are samples with the same emotion label
    but from DIFFERENT subjects (to encourage cross-subject invariance).
    Negatives are samples with different emotion labels.

    Based on SupCon (Khosla et al., 2020) with subject-aware positive mining.

    Args:
        temperature: scaling factor for similarity scores (lower = sharper)
        min_positives: minimum positive pairs required to compute loss
    """

    def __init__(self, temperature: float = 0.1, min_positives: int = 1):
        super().__init__()
        self.temperature = temperature
        self.min_positives = min_positives

    def forward(self, embeddings: torch.Tensor,
                labels: torch.Tensor,
                subjects: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (batch, embed_dim) — L2-normalized embeddings
            labels    : (batch,) — emotion class labels (0 or 1)
            subjects  : (batch,) — subject IDs

        Returns:
            scalar loss
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]

        if batch_size < 4:
            return torch.tensor(0.0, device=device)

        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Similarity matrix: (batch, batch)
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Masks
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)       # same class
        subjects_eq = subjects.unsqueeze(0) == subjects.unsqueeze(1) # same subject

        # Positives: same class, DIFFERENT subject (cross-subject invariance)
        positive_mask = labels_eq & ~subjects_eq

        # Self-mask (exclude diagonal)
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        positive_mask = positive_mask & ~self_mask

        # Check if we have enough positives
        n_positives = positive_mask.sum().item()
        if n_positives < self.min_positives:
            # Fallback: use same-class pairs (including same subject)
            positive_mask = labels_eq & ~self_mask
            n_positives = positive_mask.sum().item()
            if n_positives < self.min_positives:
                return torch.tensor(0.0, device=device)

        # For numerical stability, subtract max
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        # Denominator: all pairs except self
        neg_mask = ~self_mask
        exp_logits = torch.exp(logits) * neg_mask.float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # Mean log-probability over positive pairs
        positive_mask_float = positive_mask.float()
        mean_log_prob = (positive_mask_float * log_prob).sum(dim=1)
        n_pos_per_sample = positive_mask_float.sum(dim=1)

        # Only compute for samples that have positives
        valid = n_pos_per_sample > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device)

        mean_log_prob = mean_log_prob[valid] / n_pos_per_sample[valid]
        loss = -mean_log_prob.mean()

        return loss


class DomainAdversarialLoss(nn.Module):
    """
    Domain adversarial loss (optional, for more aggressive adaptation).

    Trains a subject discriminator on the embedding, and uses gradient
    reversal to make the embedding less subject-discriminative.

    This is a lighter alternative to full DANN — we only add it as
    a regularizer, not as the primary objective.

    Args:
        embed_dim  : dimension of the embedding
        n_subjects : number of subjects in training set
        lambda_adv : weight of the adversarial loss
    """

    def __init__(self, embed_dim: int, n_subjects: int,
                 lambda_adv: float = 0.1):
        super().__init__()
        self.lambda_adv = lambda_adv
        self.discriminator = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, n_subjects),
        )

    def forward(self, embeddings: torch.Tensor,
                subjects: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (batch, embed_dim)
            subjects  : (batch,) — subject IDs (0-indexed)

        Returns:
            adversarial loss (to be ADDED to main loss with negative sign
            via gradient reversal, or simply minimized with lambda)
        """
        # Gradient reversal: maximize subject classification error
        # Implementation: detach and negate gradients
        reversed_emb = GradientReversal.apply(embeddings, self.lambda_adv)
        logits = self.discriminator(reversed_emb)
        return F.cross_entropy(logits, subjects)


class GradientReversal(torch.autograd.Function):
    """Gradient reversal layer for domain adversarial training."""

    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_val * grad_output, None
