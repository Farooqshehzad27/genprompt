"""
Dynamic Prompt Pool for GenPrompt
Manages domain-specific visual and textual prompts.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import numpy as np


class VisualPrompt(nn.Module):
    """
    Visual prompt tokens prepended to image patches (VPT-style).
    """
    
    def __init__(
        self,
        prompt_length: int = 10,
        embed_dim: int = 512,
        init_method: str = 'random'
    ):
        """
        Args:
            prompt_length: Number of prompt tokens
            embed_dim: Embedding dimension
            init_method: Initialization method ('random', 'uniform', 'normal')
        """
        super().__init__()
        
        self.prompt_length = prompt_length
        self.embed_dim = embed_dim
        
        # Learnable prompt tokens
        self.prompt_tokens = nn.Parameter(torch.zeros(prompt_length, embed_dim))
        
        # Initialize
        self._initialize(init_method)
    
    def _initialize(self, method: str):
        """Initialize prompt tokens."""
        if method == 'random':
            nn.init.normal_(self.prompt_tokens, std=0.02)
        elif method == 'uniform':
            nn.init.uniform_(self.prompt_tokens, -0.02, 0.02)
        elif method == 'normal':
            nn.init.normal_(self.prompt_tokens, mean=0.0, std=0.02)
        else:
            raise ValueError(f"Unknown init method: {method}")
    
    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Get prompt tokens for a batch.
        
        Args:
            batch_size: Batch size
            
        Returns:
            prompt_tokens: [batch_size, prompt_length, embed_dim]
        """
        return self.prompt_tokens.unsqueeze(0).expand(batch_size, -1, -1)


class TextualPrompt(nn.Module):
    """
    Textual prompt with learnable context tokens.
    """
    
    def __init__(
        self,
        n_ctx: int = 4,
        embed_dim: int = 512,
        class_names: List[str] = ['real', 'fake'],
        init_method: str = 'random'
    ):
        """
        Args:
            n_ctx: Number of context tokens
            embed_dim: Embedding dimension
            class_names: List of class names
            init_method: Initialization method
        """
        super().__init__()
        
        self.n_ctx = n_ctx
        self.embed_dim = embed_dim
        self.class_names = class_names
        
        # Learnable context vectors
        self.ctx_vectors = nn.Parameter(torch.zeros(n_ctx, embed_dim))
        
        # Initialize
        self._initialize(init_method)
    
    def _initialize(self, method: str):
        """Initialize context vectors."""
        if method == 'random':
            nn.init.normal_(self.ctx_vectors, std=0.02)
        elif method == 'uniform':
            nn.init.uniform_(self.ctx_vectors, -0.02, 0.02)
        elif method == 'normal':
            nn.init.normal_(self.ctx_vectors, mean=0.0, std=0.02)
        else:
            raise ValueError(f"Unknown init method: {method}")
    
    def forward(self) -> torch.Tensor:
        """
        Get context vectors.
        
        Returns:
            ctx_vectors: [n_ctx, embed_dim]
        """
        return self.ctx_vectors


class PromptPool(nn.Module):
    """
    Pool of domain-specific prompts with dynamic allocation.
    """
    
    def __init__(
        self,
        visual_prompt_length: int = 10,
        textual_n_ctx: int = 4,
        embed_dim: int = 512,
        max_prompts: int = 20,
        init_method: str = 'random'
    ):
        """
        Args:
            visual_prompt_length: Length of visual prompts
            textual_n_ctx: Number of textual context tokens
            embed_dim: Embedding dimension
            max_prompts: Maximum number of prompts
            init_method: Initialization method
        """
        super().__init__()
        
        self.visual_prompt_length = visual_prompt_length
        self.textual_n_ctx = textual_n_ctx
        self.embed_dim = embed_dim
        self.max_prompts = max_prompts
        self.init_method = init_method
        
        # Prompt storage
        self.visual_prompts = nn.ModuleDict()
        self.textual_prompts = nn.ModuleDict()
        
        self.n_prompts = 0
        self.cluster_to_prompt = {}  # Maps cluster_id to prompt_id
    
    def allocate_prompt(
        self,
        cluster_id: int,
        init_from: Optional[int] = None
    ) -> int:
        """
        Allocate a new prompt for a cluster.
        
        Args:
            cluster_id: Cluster ID to allocate prompt for
            init_from: Optional cluster ID to copy initialization from
            
        Returns:
            prompt_id: ID of allocated prompt
        """
        if cluster_id in self.cluster_to_prompt:
            return self.cluster_to_prompt[cluster_id]
        
        if self.n_prompts >= self.max_prompts:
            raise ValueError(f"Maximum number of prompts ({self.max_prompts}) reached")
        
        prompt_id = self.n_prompts
        prompt_key = str(prompt_id)
        
        # Create visual prompt
        visual_prompt = VisualPrompt(
            prompt_length=self.visual_prompt_length,
            embed_dim=self.embed_dim,
            init_method=self.init_method
        )
        
        # Create textual prompt
        textual_prompt = TextualPrompt(
            n_ctx=self.textual_n_ctx,
            embed_dim=self.embed_dim,
            init_method=self.init_method
        )
        
        # Copy from existing prompt if specified
        if init_from is not None and init_from in self.cluster_to_prompt:
            source_prompt_id = self.cluster_to_prompt[init_from]
            source_key = str(source_prompt_id)
            
            if source_key in self.visual_prompts:
                visual_prompt.prompt_tokens.data.copy_(
                    self.visual_prompts[source_key].prompt_tokens.data
                )
            if source_key in self.textual_prompts:
                textual_prompt.ctx_vectors.data.copy_(
                    self.textual_prompts[source_key].ctx_vectors.data
                )
        
        # Add to pool
        self.visual_prompts[prompt_key] = visual_prompt
        self.textual_prompts[prompt_key] = textual_prompt
        
        self.cluster_to_prompt[cluster_id] = prompt_id
        self.n_prompts += 1
        
        return prompt_id
    
    def get_visual_prompt(self, prompt_id: int, batch_size: int) -> torch.Tensor:
        """Get visual prompt tokens."""
        prompt_key = str(prompt_id)
        if prompt_key not in self.visual_prompts:
            raise ValueError(f"Prompt {prompt_id} not found")
        return self.visual_prompts[prompt_key](batch_size)
    
    def get_textual_prompt(self, prompt_id: int) -> torch.Tensor:
        """Get textual prompt context vectors."""
        prompt_key = str(prompt_id)
        if prompt_key not in self.textual_prompts:
            raise ValueError(f"Prompt {prompt_id} not found")
        return self.textual_prompts[prompt_key]()
    
    def get_all_visual_prompts(self, batch_size: int) -> torch.Tensor:
        """
        Get all visual prompts stacked.
        
        Returns:
            prompts: [n_prompts, batch_size, prompt_length, embed_dim]
        """
        prompts = []
        for i in range(self.n_prompts):
            prompt = self.get_visual_prompt(i, batch_size)
            prompts.append(prompt)
        return torch.stack(prompts, dim=0)
    
    def get_prompt_for_cluster(self, cluster_id: int) -> Optional[int]:
        """Get prompt ID for a cluster."""
        return self.cluster_to_prompt.get(cluster_id, None)
    
    def get_statistics(self) -> Dict:
        """Get prompt pool statistics."""
        return {
            'n_prompts': self.n_prompts,
            'max_prompts': self.max_prompts,
            'visual_prompt_length': self.visual_prompt_length,
            'textual_n_ctx': self.textual_n_ctx,
            'cluster_to_prompt': self.cluster_to_prompt
        }


if __name__ == "__main__":
    # Test prompt pool
    print("Testing PromptPool...")
    
    pool = PromptPool(
        visual_prompt_length=10,
        textual_n_ctx=4,
        embed_dim=512,
        max_prompts=20
    )
    
    # Allocate prompts for clusters
    print("\nAllocating prompts...")
    for cluster_id in range(5):
        prompt_id = pool.allocate_prompt(cluster_id)
        print(f"  Cluster {cluster_id} -> Prompt {prompt_id}")
    
    # Get prompts
    print("\nGetting prompts...")
    batch_size = 32
    
    visual_prompt = pool.get_visual_prompt(0, batch_size)
    print(f"  Visual prompt shape: {visual_prompt.shape}")
    
    textual_prompt = pool.get_textual_prompt(0)
    print(f"  Textual prompt shape: {textual_prompt.shape}")
    
    all_visual = pool.get_all_visual_prompts(batch_size)
    print(f"  All visual prompts shape: {all_visual.shape}")
    
    # Statistics
    stats = pool.get_statistics()
    print(f"\nPrompt Pool Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✓ PromptPool tests passed!")
