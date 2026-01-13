"""
Prompt Routing Network (PRN)
Routes images to appropriate domain-specific prompts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class PromptRouter(nn.Module):
    """
    Lightweight MLP router for prompt selection.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        n_prompts: int = 12,
        dropout: float = 0.1,
        routing_mode: str = 'soft'
    ):
        """
        Args:
            input_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension
            n_prompts: Number of prompts to route to
            dropout: Dropout rate
            routing_mode: 'soft' (mixture-of-experts) or 'hard' (argmax)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_prompts = n_prompts
        self.routing_mode = routing_mode
        
        # MLP router
        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_prompts)
        )
        
        # Initialize
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize router weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        embeddings: torch.Tensor,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route embeddings to prompts.
        
        Args:
            embeddings: Input embeddings [batch_size, input_dim]
            temperature: Temperature for softmax (lower = sharper routing)
            
        Returns:
            routing_weights: [batch_size, n_prompts]
            routing_logits: [batch_size, n_prompts]
        """
        # Get routing logits
        logits = self.router(embeddings)
        
        # Apply temperature
        logits = logits / temperature
        
        # Compute routing weights
        if self.routing_mode == 'soft':
            # Soft routing: weighted combination
            weights = F.softmax(logits, dim=-1)
        elif self.routing_mode == 'hard':
            # Hard routing: one-hot argmax
            indices = torch.argmax(logits, dim=-1)
            weights = F.one_hot(indices, num_classes=self.n_prompts).float()
        else:
            raise ValueError(f"Unknown routing mode: {self.routing_mode}")
        
        return weights, logits
    
    def get_top_k_routes(
        self,
        embeddings: torch.Tensor,
        k: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get top-k routing weights.
        
        Args:
            embeddings: Input embeddings [batch_size, input_dim]
            k: Number of top prompts to select
            
        Returns:
            top_k_weights: [batch_size, k]
            top_k_indices: [batch_size, k]
        """
        logits = self.router(embeddings)
        weights = F.softmax(logits, dim=-1)
        
        top_k_weights, top_k_indices = torch.topk(weights, k, dim=-1)
        
        return top_k_weights, top_k_indices


class PromptRoutingNetwork(nn.Module):
    """
    Complete prompt routing network with classifier head.
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 256,
        n_prompts: int = 12,
        n_classes: int = 2,
        dropout: float = 0.1,
        routing_mode: str = 'soft'
    ):
        """
        Args:
            embed_dim: Embedding dimension
            hidden_dim: Hidden dimension for router
            n_prompts: Number of prompts
            n_classes: Number of classes (2 for real/fake)
            dropout: Dropout rate
            routing_mode: 'soft' or 'hard'
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.n_prompts = n_prompts
        self.n_classes = n_classes
        self.routing_mode = routing_mode
        
        # Prompt router
        self.router = PromptRouter(
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_prompts=n_prompts,
            dropout=dropout,
            routing_mode=routing_mode
        )
        
        # Per-prompt classifiers (prompt experts)
        self.prompt_classifiers = nn.ModuleList([
            nn.Linear(embed_dim, n_classes)
            for _ in range(n_prompts)
        ])
        
        # Initialize classifiers
        for classifier in self.prompt_classifiers:
            nn.init.xavier_uniform_(classifier.weight)
            nn.init.constant_(classifier.bias, 0)
    
    def forward(
        self,
        embeddings: torch.Tensor,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with prompt routing.
        
        Args:
            embeddings: Input embeddings [batch_size, embed_dim]
            temperature: Routing temperature
            
        Returns:
            logits: Final classification logits [batch_size, n_classes]
            routing_weights: Routing weights [batch_size, n_prompts]
            expert_logits: Per-expert logits [batch_size, n_prompts, n_classes]
        """
        batch_size = embeddings.size(0)
        
        # Get routing weights
        routing_weights, routing_logits = self.router(embeddings, temperature)
        
        # Get predictions from each prompt expert
        expert_logits = []
        for i, classifier in enumerate(self.prompt_classifiers):
            expert_logit = classifier(embeddings)  # [batch_size, n_classes]
            expert_logits.append(expert_logit)
        
        expert_logits = torch.stack(expert_logits, dim=1)  # [batch_size, n_prompts, n_classes]
        
        # Combine expert predictions using routing weights
        if self.routing_mode == 'soft':
            # Weighted combination
            routing_weights_expanded = routing_weights.unsqueeze(-1)  # [batch_size, n_prompts, 1]
            logits = (expert_logits * routing_weights_expanded).sum(dim=1)  # [batch_size, n_classes]
        else:
            # Hard routing: use only selected expert
            selected_indices = torch.argmax(routing_weights, dim=-1)  # [batch_size]
            logits = expert_logits[torch.arange(batch_size), selected_indices]  # [batch_size, n_classes]
        
        return logits, routing_weights, expert_logits
    
    def add_prompt_expert(self):
        """Add a new prompt expert (for dynamic prompt allocation)."""
        new_classifier = nn.Linear(self.embed_dim, self.n_classes)
        nn.init.xavier_uniform_(new_classifier.weight)
        nn.init.constant_(new_classifier.bias, 0)
        
        # Move to same device as existing classifiers
        if len(self.prompt_classifiers) > 0:
            device = self.prompt_classifiers[0].weight.device
            new_classifier = new_classifier.to(device)
        
        self.prompt_classifiers.append(new_classifier)
        self.n_prompts += 1
        
        # Update router output dimension
        # Note: This requires reinitializing the router's final layer
        old_router_final = self.router.router[-1]
        new_router_final = nn.Linear(old_router_final.in_features, self.n_prompts)
        
        # Copy old weights
        with torch.no_grad():
            new_router_final.weight[:self.n_prompts-1] = old_router_final.weight
            new_router_final.bias[:self.n_prompts-1] = old_router_final.bias
            # Initialize new prompt routing weights
            nn.init.xavier_uniform_(new_router_final.weight[-1:])
            nn.init.constant_(new_router_final.bias[-1:], 0)
        
        if len(self.prompt_classifiers) > 0:
            device = self.prompt_classifiers[0].weight.device
            new_router_final = new_router_final.to(device)
        
        self.router.router[-1] = new_router_final
        self.router.n_prompts = self.n_prompts


def compute_routing_diversity(routing_weights: torch.Tensor) -> float:
    """
    Compute routing diversity (entropy).
    Higher entropy = more diverse routing.
    
    Args:
        routing_weights: [batch_size, n_prompts]
        
    Returns:
        diversity: Average entropy
    """
    # Compute entropy for each sample
    eps = 1e-10
    entropy = -(routing_weights * torch.log(routing_weights + eps)).sum(dim=-1)
    return entropy.mean().item()


def compute_routing_concentration(routing_weights: torch.Tensor) -> float:
    """
    Compute routing concentration (how much weight goes to top prompt).
    
    Args:
        routing_weights: [batch_size, n_prompts]
        
    Returns:
        concentration: Average max weight
    """
    max_weights = routing_weights.max(dim=-1)[0]
    return max_weights.mean().item()


if __name__ == "__main__":
    # Test prompt routing network
    print("Testing PromptRoutingNetwork...")
    
    # Create network
    prn = PromptRoutingNetwork(
        embed_dim=512,
        hidden_dim=256,
        n_prompts=12,
        n_classes=2,
        routing_mode='soft'
    )
    
    print(f"Created PRN with {prn.n_prompts} prompts")
    
    # Test forward pass
    batch_size = 32
    embeddings = torch.randn(batch_size, 512)
    
    logits, routing_weights, expert_logits = prn(embeddings, temperature=1.0)
    
    print(f"\nForward pass:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Routing weights shape: {routing_weights.shape}")
    print(f"  Expert logits shape: {expert_logits.shape}")
    
    # Check routing weights sum to 1
    weights_sum = routing_weights.sum(dim=-1)
    print(f"  Routing weights sum: {weights_sum[0]:.4f} (should be ~1.0)")
    
    # Compute routing statistics
    diversity = compute_routing_diversity(routing_weights)
    concentration = compute_routing_concentration(routing_weights)
    
    print(f"\nRouting statistics:")
    print(f"  Diversity (entropy): {diversity:.4f}")
    print(f"  Concentration (max weight): {concentration:.4f}")
    
    # Test adding new prompt expert
    print(f"\nAdding new prompt expert...")
    prn.add_prompt_expert()
    print(f"  New number of prompts: {prn.n_prompts}")
    
    # Test with new prompt
    logits, routing_weights, expert_logits = prn(embeddings, temperature=1.0)
    print(f"  New routing weights shape: {routing_weights.shape}")
    print(f"  New expert logits shape: {expert_logits.shape}")
    
    # Test hard routing
    print(f"\nTesting hard routing...")
    prn_hard = PromptRoutingNetwork(
        embed_dim=512,
        hidden_dim=256,
        n_prompts=12,
        n_classes=2,
        routing_mode='hard'
    )
    
    logits, routing_weights, expert_logits = prn_hard(embeddings, temperature=1.0)
    print(f"  Hard routing weights (first sample): {routing_weights[0]}")
    print(f"  Sum: {routing_weights[0].sum():.4f}")
    
    print("\n✓ PromptRoutingNetwork tests passed!")
