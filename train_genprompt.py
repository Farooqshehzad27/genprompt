"""
GenPrompt Training Pipeline
End-to-end training for unsupervised domain discovery and prompt routing.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import our modules
from data_loader import CDDBDataset, ContinualCDDBDataset, get_clip_transforms
from clip_encoder import CLIPEncoder
from domain_discovery import DomainDiscovery, ClusterEvaluator
from prompt_pool import PromptPool
from prompt_router import PromptRoutingNetwork
from contrastive_loss import PrototypeContrastiveLearning


class GenPromptTrainer:
    """
    Complete GenPrompt training pipeline.
    """
    
    def __init__(
        self,
        data_root: str,
        domain_order: List[str],
        clip_model: str = 'ViT-B/32',
        n_clusters: int = 12,
        clustering_method: str = 'kmeans',
        visual_prompt_length: int = 10,
        textual_n_ctx: int = 4,
        hidden_dim: int = 256,
        routing_mode: str = 'soft',
        temperature: float = 0.07,
        lambda_cls: float = 1.0,
        lambda_con: float = 0.5,
        lr_prompts: float = 1e-3,
        lr_router: float = 1e-4,
        batch_size: int = 64,
        epochs_per_domain: int = 10,
        device: str = 'cuda',
        output_dir: str = 'experiments',
        cache_dir: str = 'clip_embeddings',
        use_cached_embeddings: bool = True
    ):
        """
        Args:
            data_root: Path to CDDB dataset
            domain_order: Ordered list of domains for continual learning
            clip_model: CLIP model name
            n_clusters: Number of clusters for domain discovery
            clustering_method: Clustering algorithm
            visual_prompt_length: Length of visual prompts
            textual_n_ctx: Number of textual context tokens
            hidden_dim: Hidden dimension for router
            routing_mode: 'soft' or 'hard'
            temperature: Temperature for contrastive loss
            lambda_cls: Weight for classification loss
            lambda_con: Weight for contrastive loss
            lr_prompts: Learning rate for prompts
            lr_router: Learning rate for router
            batch_size: Batch size
            epochs_per_domain: Training epochs per domain
            device: Device to train on
            output_dir: Output directory
            cache_dir: Directory for cached embeddings
            use_cached_embeddings: Whether to use cached embeddings
        """
        self.data_root = data_root
        self.domain_order = domain_order
        self.clip_model = clip_model
        self.n_clusters = n_clusters
        self.clustering_method = clustering_method
        self.batch_size = batch_size
        self.epochs_per_domain = epochs_per_domain
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.use_cached_embeddings = use_cached_embeddings
        
        # Loss weights
        self.lambda_cls = lambda_cls
        self.lambda_con = lambda_con
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        print("Initializing GenPrompt components...")
        
        # CLIP encoder
        self.clip_encoder = CLIPEncoder(
            model_name=clip_model,
            device=str(self.device),
            cache_dir=str(self.cache_dir)
        )
        embed_dim = self.clip_encoder.embed_dim
        
        # Domain discovery
        self.domain_discoverer = DomainDiscovery(
            method=clustering_method,
            n_clusters=n_clusters,
            random_state=42
        )
        
        # Prompt pool
        self.prompt_pool = PromptPool(
            visual_prompt_length=visual_prompt_length,
            textual_n_ctx=textual_n_ctx,
            embed_dim=embed_dim,
            max_prompts=n_clusters * 2  # Allow some flexibility
        ).to(self.device)
        
        # Prompt routing network
        self.prn = PromptRoutingNetwork(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_prompts=n_clusters,
            n_classes=2,
            routing_mode=routing_mode
        ).to(self.device)
        
        # Contrastive learning
        self.contrastive_learner = PrototypeContrastiveLearning(
            embed_dim=embed_dim,
            temperature=temperature,
            prototype_momentum=0.9
        ).to(self.device)
        
        # Optimizers
        self.optimizer_prompts = torch.optim.Adam(
            self.prompt_pool.parameters(),
            lr=lr_prompts
        )
        self.optimizer_router = torch.optim.Adam(
            self.prn.parameters(),
            lr=lr_router
        )
        
        # Training state
        self.current_domain_idx = 0
        self.seen_domains = []
        self.cluster_to_domain = {}  # Track which clusters belong to which domain
        self.training_history = []
        
        print(f"✓ GenPrompt initialized on {self.device}")
        print(f"  CLIP model: {clip_model} (embed_dim={embed_dim})")
        print(f"  Clustering: {clustering_method} (K={n_clusters})")
        print(f"  Routing mode: {routing_mode}")
    
    def extract_embeddings(
        self,
        dataset: CDDBDataset,
        cache_name: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """Extract CLIP embeddings for a dataset."""
        return self.clip_encoder.extract_dataset_embeddings(
            dataset=dataset,
            batch_size=self.batch_size * 2,  # Larger batch for extraction
            num_workers=4,
            cache_name=cache_name,
            force_recompute=not self.use_cached_embeddings
        )
    
    def discover_domains(
        self,
        embeddings: np.ndarray,
        domain_name: str
    ) -> np.ndarray:
        """
        Discover pseudo-domains in embeddings.
        
        Args:
            embeddings: CLIP embeddings
            domain_name: Name of the domain (for tracking)
            
        Returns:
            cluster_assignments: Cluster labels
        """
        print(f"\nDiscovering pseudo-domains in {domain_name}...")
        
        cluster_assignments = self.domain_discoverer.fit(embeddings)
        
        # Track which clusters belong to this domain
        unique_clusters = np.unique(cluster_assignments)
        unique_clusters = unique_clusters[unique_clusters >= 0]
        
        for cluster_id in unique_clusters:
            if cluster_id not in self.cluster_to_domain:
                self.cluster_to_domain[cluster_id] = domain_name
        
        return cluster_assignments
    
    def allocate_prompts(self, cluster_assignments: np.ndarray):
        """Allocate prompts for discovered clusters."""
        unique_clusters = np.unique(cluster_assignments)
        unique_clusters = unique_clusters[unique_clusters >= 0]
        
        print(f"Allocating prompts for {len(unique_clusters)} clusters...")
        
        for cluster_id in unique_clusters:
            cluster_id = int(cluster_id)
            
            # Check if prompt already exists
            if self.prompt_pool.get_prompt_for_cluster(cluster_id) is None:
                # Allocate new prompt
                prompt_id = self.prompt_pool.allocate_prompt(cluster_id)
                
                # Add corresponding expert to PRN
                if prompt_id >= self.prn.n_prompts:
                    self.prn.add_prompt_expert()
                
                print(f"  Cluster {cluster_id} -> Prompt {prompt_id} (new)")
            else:
                prompt_id = self.prompt_pool.get_prompt_for_cluster(cluster_id)
                print(f"  Cluster {cluster_id} -> Prompt {prompt_id} (existing)")
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        embeddings: torch.Tensor,
        cluster_assignments: torch.Tensor,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.prompt_pool.train()
        self.prn.train()
        
        total_loss = 0
        total_cls_loss = 0
        total_con_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Get CLIP embeddings for this batch
            batch_embeddings = self.clip_encoder.encode_images(images, normalize=True)
            
            # Get cluster assignments for this batch
            start_idx = batch_idx * self.batch_size
            end_idx = start_idx + len(images)
            batch_clusters = cluster_assignments[start_idx:end_idx]
            
            # Forward pass through PRN
            logits, routing_weights, expert_logits = self.prn(batch_embeddings)
            
            # Classification loss
            cls_loss = F.cross_entropy(logits, labels)
            
            # Contrastive loss
            con_loss = self.contrastive_learner.compute_loss(
                batch_embeddings,
                batch_clusters
            )
            
            # Total loss
            loss = self.lambda_cls * cls_loss + self.lambda_con * con_loss
            
            # Backward pass
            self.optimizer_prompts.zero_grad()
            self.optimizer_router.zero_grad()
            loss.backward()
            self.optimizer_prompts.step()
            self.optimizer_router.step()
            
            # Update prototypes
            self.contrastive_learner.update_prototypes(
                batch_embeddings.detach(),
                batch_clusters
            )
            
            # Statistics
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_con_loss += con_loss.item()
            
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += len(labels)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cls': f'{cls_loss.item():.4f}',
                'con': f'{con_loss.item():.4f}',
                'acc': f'{100*correct/total:.2f}%'
            })
        
        return {
            'loss': total_loss / len(dataloader),
            'cls_loss': total_cls_loss / len(dataloader),
            'con_loss': total_con_loss / len(dataloader),
            'accuracy': 100 * correct / total
        }
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        embeddings: torch.Tensor,
        cluster_assignments: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.prompt_pool.eval()
        self.prn.eval()
        
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Get CLIP embeddings
            batch_embeddings = self.clip_encoder.encode_images(images, normalize=True)
            
            # Forward pass
            logits, routing_weights, expert_logits = self.prn(batch_embeddings)
            
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += len(labels)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        accuracy = 100 * correct / total
        
        return {
            'accuracy': accuracy,
            'predictions': np.array(all_preds),
            'labels': np.array(all_labels)
        }
    
    def train_domain(self, domain_name: str):
        """Train on a single domain."""
        print(f"\n{'='*80}")
        print(f"Training on domain: {domain_name}")
        print(f"{'='*80}")
        
        # Create dataset
        train_dataset = CDDBDataset(
            data_root=self.data_root,
            domains=[domain_name],
            split='train',
            transform=get_clip_transforms(),
            return_domain_label=False
        )
        
        val_dataset = CDDBDataset(
            data_root=self.data_root,
            domains=[domain_name],
            split='val',
            transform=get_clip_transforms(),
            return_domain_label=False
        )
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        
        # Extract embeddings
        cache_name = f"{domain_name}_train_{self.clip_model.replace('/', '_')}"
        train_data = self.extract_embeddings(train_dataset, cache_name)
        
        train_embeddings = torch.from_numpy(train_data['embeddings']).float()
        train_labels = torch.from_numpy(train_data['labels']).long()
        
        # Discover domains
        cluster_assignments = self.discover_domains(
            train_data['embeddings'],
            domain_name
        )
        cluster_assignments = torch.from_numpy(cluster_assignments).long()
        
        # Allocate prompts
        self.allocate_prompts(cluster_assignments.numpy())
        
        # Create dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Training loop
        for epoch in range(self.epochs_per_domain):
            metrics = self.train_epoch(
                train_loader,
                train_embeddings,
                cluster_assignments,
                epoch
            )
            
            print(f"\nEpoch {epoch+1}/{self.epochs_per_domain}")
            print(f"  Loss: {metrics['loss']:.4f}")
            print(f"  Cls Loss: {metrics['cls_loss']:.4f}")
            print(f"  Con Loss: {metrics['con_loss']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']:.2f}%")
        
        # Save checkpoint
        self.save_checkpoint(domain_name)
        
        # Track domain
        self.seen_domains.append(domain_name)
        self.current_domain_idx += 1
    
    def save_checkpoint(self, domain_name: str):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f'checkpoint_{domain_name}.pt'
        
        torch.save({
            'domain_name': domain_name,
            'seen_domains': self.seen_domains,
            'prompt_pool': self.prompt_pool.state_dict(),
            'prn': self.prn.state_dict(),
            'cluster_to_domain': self.cluster_to_domain,
            'optimizer_prompts': self.optimizer_prompts.state_dict(),
            'optimizer_router': self.optimizer_router.state_dict()
        }, checkpoint_path)
        
        print(f"\n✓ Checkpoint saved: {checkpoint_path}")
    
    def run(self):
        """Run complete training pipeline."""
        print(f"\nStarting GenPrompt training on {len(self.domain_order)} domains")
        print(f"Domain order: {', '.join(self.domain_order)}")
        
        for domain_name in self.domain_order:
            self.train_domain(domain_name)
        
        print(f"\n{'='*80}")
        print("✓ Training complete!")
        print(f"{'='*80}")
        print(f"Trained on {len(self.seen_domains)} domains")
        print(f"Final number of prompts: {self.prompt_pool.n_prompts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GenPrompt on CDDB dataset')
    
    # Data
    parser.add_argument('--data_root', type=str, default=r'd:\Farooq\genprompt\CDDB',
                       help='Path to CDDB dataset')
    parser.add_argument('--domains', type=str, nargs='+',
                       default=['biggan', 'crn', 'cyclegan', 'deepfake'],
                       help='Domains to train on (in order)')
    
    # Model
    parser.add_argument('--clip_model', type=str, default='ViT-B/32',
                       choices=['ViT-B/32', 'ViT-L/14'],
                       help='CLIP model')
    parser.add_argument('--n_clusters', type=int, default=12,
                       help='Number of clusters')
    parser.add_argument('--clustering_method', type=str, default='kmeans',
                       choices=['kmeans', 'dbscan', 'agglomerative'],
                       help='Clustering method')
    parser.add_argument('--routing_mode', type=str, default='soft',
                       choices=['soft', 'hard'],
                       help='Routing mode')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--epochs_per_domain', type=int, default=10,
                       help='Epochs per domain')
    parser.add_argument('--lr_prompts', type=float, default=1e-3,
                       help='Learning rate for prompts')
    parser.add_argument('--lr_router', type=float, default=1e-4,
                       help='Learning rate for router')
    parser.add_argument('--lambda_cls', type=float, default=1.0,
                       help='Weight for classification loss')
    parser.add_argument('--lambda_con', type=float, default=0.5,
                       help='Weight for contrastive loss')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='experiments/genprompt',
                       help='Output directory')
    parser.add_argument('--cache_dir', type=str, default='clip_embeddings',
                       help='Cache directory for embeddings')
    parser.add_argument('--use_cached', action='store_true',
                       help='Use cached embeddings if available')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to train on')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = GenPromptTrainer(
        data_root=args.data_root,
        domain_order=args.domains,
        clip_model=args.clip_model,
        n_clusters=args.n_clusters,
        clustering_method=args.clustering_method,
        routing_mode=args.routing_mode,
        batch_size=args.batch_size,
        epochs_per_domain=args.epochs_per_domain,
        lr_prompts=args.lr_prompts,
        lr_router=args.lr_router,
        lambda_cls=args.lambda_cls,
        lambda_con=args.lambda_con,
        device=args.device,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        use_cached_embeddings=args.use_cached
    )
    
    # Run training
    trainer.run()
