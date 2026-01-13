"""
Cluster-Aligned Contrastive Learning
Stabilizes prompt learning and prevents catastrophic forgetting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import numpy as np


class ClusterPrototypes:
    """
    Maintains cluster prototypes with exponential moving average updates.
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        momentum: float = 0.9
    ):
        """
        Args:
            embed_dim: Embedding dimension
            momentum: Momentum for EMA updates
        """
        self.embed_dim = embed_dim
        self.momentum = momentum
        
        self.prototypes = {}  # cluster_id -> prototype tensor
    
    def update(
        self,
        cluster_id: int,
        embeddings: torch.Tensor,
        device: Optional[torch.device] = None
    ):
        """
        Update prototype for a cluster.
        
        Args:
            cluster_id: Cluster ID
            embeddings: Embeddings from this cluster [N, embed_dim]
            device: Device to store prototypes on
        """
        # Compute mean of embeddings
        new_prototype = embeddings.mean(dim=0)
        
        # L2 normalize
        new_prototype = F.normalize(new_prototype, p=2, dim=0)
        
        if cluster_id in self.prototypes:
            # EMA update
            old_prototype = self.prototypes[cluster_id]
            updated_prototype = self.momentum * old_prototype + (1 - self.momentum) * new_prototype
            updated_prototype = F.normalize(updated_prototype, p=2, dim=0)
            self.prototypes[cluster_id] = updated_prototype
        else:
            # Initialize new prototype
            self.prototypes[cluster_id] = new_prototype
    
    def get_prototype(self, cluster_id: int) -> Optional[torch.Tensor]:
        """Get prototype for a cluster."""
        return self.prototypes.get(cluster_id, None)
    
    def get_all_prototypes(self) -> torch.Tensor:
        """
        Get all prototypes as a tensor.
        
        Returns:
            prototypes: [n_clusters, embed_dim]
        """
        if len(self.prototypes) == 0:
            return None
        
        cluster_ids = sorted(self.prototypes.keys())
        prototypes = torch.stack([self.prototypes[cid] for cid in cluster_ids], dim=0)
        return prototypes
    
    def to(self, device: torch.device):
        """Move all prototypes to device."""
        for cluster_id in self.prototypes:
            self.prototypes[cluster_id] = self.prototypes[cluster_id].to(device)
        return self


class ContrastiveLoss(nn.Module):
    """
    Cluster-aligned contrastive loss for prompt learning.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07
    ):
        """
        Args:
            temperature: Temperature parameter for scaling
            base_temperature: Base temperature for normalization
        """
        super().__init__()
        
        self.temperature = temperature
        self.base_temperature = base_temperature
    
    def forward(
        self,
        embeddings: torch.Tensor,
        cluster_assignments: torch.Tensor,
        prototypes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            embeddings: Input embeddings [batch_size, embed_dim]
            cluster_assignments: Cluster labels [batch_size]
            prototypes: Cluster prototypes [n_clusters, embed_dim]
            
        Returns:
            loss: Contrastive loss value
        """
        batch_size = embeddings.size(0)
        device = embeddings.device
        
        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity to all prototypes
        # [batch_size, n_clusters]
        similarities = torch.matmul(embeddings, prototypes.t()) / self.temperature
        
        # Create mask for positive pairs (same cluster)
        # [batch_size, n_clusters]
        cluster_ids = torch.arange(prototypes.size(0), device=device)
        positive_mask = cluster_assignments.unsqueeze(1) == cluster_ids.unsqueeze(0)
        
        # Compute log probabilities
        log_prob = F.log_softmax(similarities, dim=1)
        
        # Compute mean of log-likelihood over positive pairs
        loss = -(positive_mask * log_prob).sum(dim=1) / positive_mask.sum(dim=1).clamp(min=1)
        loss = loss.mean()
        
        return loss


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning loss (for comparison).
    From: https://arxiv.org/abs/2004.11362
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07
    ):
        """
        Args:
            temperature: Temperature parameter
            base_temperature: Base temperature
        """
        super().__init__()
        
        self.temperature = temperature
        self.base_temperature = base_temperature
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.
        
        Args:
            features: Feature embeddings [batch_size, embed_dim]
            labels: Ground truth labels [batch_size]
            
        Returns:
            loss: SupCon loss value
        """
        device = features.device
        batch_size = features.size(0)
        
        # L2 normalize
        features = F.normalize(features, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.t()) / self.temperature
        
        # Create mask for positive pairs (same label)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.t()).float().to(device)
        
        # Mask out self-similarity
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Compute log probabilities
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True))
        
        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        
        return loss


class PrototypeContrastiveLearning:
    """
    Complete prototype-based contrastive learning module.
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        temperature: float = 0.07,
        prototype_momentum: float = 0.9
    ):
        """
        Args:
            embed_dim: Embedding dimension
            temperature: Temperature for contrastive loss
            prototype_momentum: Momentum for prototype updates
        """
        self.embed_dim = embed_dim
        self.temperature = temperature
        
        # Cluster prototypes
        self.prototypes = ClusterPrototypes(
            embed_dim=embed_dim,
            momentum=prototype_momentum
        )
        
        # Contrastive loss
        self.criterion = ContrastiveLoss(temperature=temperature)
    
    def update_prototypes(
        self,
        embeddings: torch.Tensor,
        cluster_assignments: torch.Tensor
    ):
        """
        Update cluster prototypes with new embeddings.
        
        Args:
            embeddings: CLIP embeddings [batch_size, embed_dim]
            cluster_assignments: Cluster labels [batch_size]
        """
        unique_clusters = torch.unique(cluster_assignments)
        
        for cluster_id in unique_clusters:
            if cluster_id < 0:  # Skip noise
                continue
            
            mask = cluster_assignments == cluster_id
            cluster_embeddings = embeddings[mask]
            
            self.prototypes.update(
                cluster_id=cluster_id.item(),
                embeddings=cluster_embeddings,
                device=embeddings.device
            )
    
    def compute_loss(
        self,
        embeddings: torch.Tensor,
        cluster_assignments: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            embeddings: CLIP embeddings [batch_size, embed_dim]
            cluster_assignments: Cluster labels [batch_size]
            
        Returns:
            loss: Contrastive loss value
        """
        # Get all prototypes
        prototypes = self.prototypes.get_all_prototypes()
        
        if prototypes is None:
            return torch.tensor(0.0, device=embeddings.device)
        
        # Move prototypes to same device as embeddings
        prototypes = prototypes.to(embeddings.device)
        
        # Compute loss
        loss = self.criterion(embeddings, cluster_assignments, prototypes)
        
        return loss
    
    def to(self, device: torch.device):
        """Move to device."""
        self.prototypes.to(device)
        return self


def compute_prototype_alignment(
    embeddings: torch.Tensor,
    cluster_assignments: torch.Tensor,
    prototypes: torch.Tensor
) -> Dict[str, float]:
    """
    Compute alignment metrics between embeddings and prototypes.
    
    Args:
        embeddings: CLIP embeddings [batch_size, embed_dim]
        cluster_assignments: Cluster labels [batch_size]
        prototypes: Cluster prototypes [n_clusters, embed_dim]
        
    Returns:
        metrics: Dictionary of alignment metrics
    """
    embeddings = F.normalize(embeddings, p=2, dim=1)
    prototypes = F.normalize(prototypes, p=2, dim=1)
    
    # Compute similarity to assigned prototype
    assigned_prototypes = prototypes[cluster_assignments]
    alignment = (embeddings * assigned_prototypes).sum(dim=1)
    
    # Compute similarity to all prototypes
    all_similarities = torch.matmul(embeddings, prototypes.t())
    
    metrics = {
        'mean_alignment': alignment.mean().item(),
        'min_alignment': alignment.min().item(),
        'max_alignment': alignment.max().item(),
        'mean_max_similarity': all_similarities.max(dim=1)[0].mean().item()
    }
    
    return metrics


if __name__ == "__main__":
    # Test contrastive learning module
    print("Testing PrototypeContrastiveLearning...")
    
    # Create module
    pcl = PrototypeContrastiveLearning(
        embed_dim=512,
        temperature=0.07,
        prototype_momentum=0.9
    )
    
    # Simulate embeddings and cluster assignments
    batch_size = 64
    n_clusters = 5
    
    embeddings = torch.randn(batch_size, 512)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    cluster_assignments = torch.randint(0, n_clusters, (batch_size,))
    
    print(f"\nBatch size: {batch_size}")
    print(f"Number of clusters: {n_clusters}")
    
    # Update prototypes
    print("\nUpdating prototypes...")
    pcl.update_prototypes(embeddings, cluster_assignments)
    
    prototypes = pcl.prototypes.get_all_prototypes()
    print(f"  Prototypes shape: {prototypes.shape}")
    
    # Compute loss
    print("\nComputing contrastive loss...")
    loss = pcl.compute_loss(embeddings, cluster_assignments)
    print(f"  Loss: {loss.item():.4f}")
    
    # Compute alignment metrics
    print("\nComputing alignment metrics...")
    metrics = compute_prototype_alignment(embeddings, cluster_assignments, prototypes)
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Test with new batch (simulating online learning)
    print("\nTesting with new batch...")
    new_embeddings = torch.randn(32, 512)
    new_embeddings = F.normalize(new_embeddings, p=2, dim=1)
    new_assignments = torch.randint(0, n_clusters, (32,))
    
    pcl.update_prototypes(new_embeddings, new_assignments)
    loss = pcl.compute_loss(new_embeddings, new_assignments)
    print(f"  New loss: {loss.item():.4f}")
    
    # Test SupConLoss
    print("\nTesting SupConLoss...")
    supcon = SupConLoss(temperature=0.07)
    
    features = torch.randn(64, 512)
    labels = torch.randint(0, 2, (64,))  # Binary labels (real/fake)
    
    loss = supcon(features, labels)
    print(f"  SupCon loss: {loss.item():.4f}")
    
    print("\n✓ Contrastive learning tests passed!")
