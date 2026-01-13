"""
CLIP Feature Extraction Module
Extracts and caches CLIP embeddings for CDDB dataset.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import clip

from data_loader import CDDBDataset, get_clip_transforms


class CLIPEncoder:
    """
    CLIP feature extractor with caching support.
    """
    
    def __init__(
        self,
        model_name: str = 'ViT-B/32',
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Args:
            model_name: CLIP model name ('ViT-B/32' or 'ViT-L/14')
            device: Device to run on (cuda/cpu)
            cache_dir: Directory to cache embeddings
        """
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load CLIP model
        print(f"Loading CLIP model: {model_name}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        
        # Get embedding dimension
        self.embed_dim = self.model.visual.output_dim
        
        print(f"CLIP model loaded on {self.device}")
        print(f"Embedding dimension: {self.embed_dim}")
    
    @torch.no_grad()
    def encode_images(
        self,
        images: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Encode images to CLIP embeddings.
        
        Args:
            images: Batch of images [B, 3, H, W]
            normalize: Whether to L2-normalize embeddings
            
        Returns:
            embeddings: [B, embed_dim]
        """
        images = images.to(self.device)
        embeddings = self.model.encode_image(images)
        
        if normalize:
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        return embeddings
    
    @torch.no_grad()
    def encode_text(
        self,
        texts: List[str],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Encode text to CLIP embeddings.
        
        Args:
            texts: List of text strings
            normalize: Whether to L2-normalize embeddings
            
        Returns:
            embeddings: [len(texts), embed_dim]
        """
        text_tokens = clip.tokenize(texts).to(self.device)
        embeddings = self.model.encode_text(text_tokens)
        
        if normalize:
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        return embeddings
    
    def extract_dataset_embeddings(
        self,
        dataset: CDDBDataset,
        batch_size: int = 128,
        num_workers: int = 4,
        cache_name: Optional[str] = None,
        force_recompute: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Extract CLIP embeddings for entire dataset.
        
        Args:
            dataset: CDDB dataset
            batch_size: Batch size for extraction
            num_workers: Number of data loading workers
            cache_name: Name for cached embeddings file
            force_recompute: Force recomputation even if cache exists
            
        Returns:
            Dictionary with 'embeddings', 'labels', 'domain_indices'
        """
        # Check cache
        if cache_name and self.cache_dir and not force_recompute:
            cache_path = self.cache_dir / f"{cache_name}.npz"
            if cache_path.exists():
                print(f"Loading cached embeddings from {cache_path}")
                data = np.load(cache_path)
                return {
                    'embeddings': data['embeddings'],
                    'labels': data['labels'],
                    'domain_indices': data['domain_indices']
                }
        
        # Extract embeddings
        print(f"Extracting CLIP embeddings for {len(dataset)} samples...")
        
        # Temporarily set dataset to return domain labels
        original_return_domain = dataset.return_domain_label
        dataset.return_domain_label = True
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        all_embeddings = []
        all_labels = []
        all_domain_indices = []
        
        for images, labels, domain_indices in tqdm(loader, desc="Extracting embeddings"):
            embeddings = self.encode_images(images, normalize=True)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())
            all_domain_indices.append(domain_indices.numpy())
        
        # Restore original setting
        dataset.return_domain_label = original_return_domain
        
        # Concatenate
        embeddings = np.concatenate(all_embeddings, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        domain_indices = np.concatenate(all_domain_indices, axis=0)
        
        result = {
            'embeddings': embeddings,
            'labels': labels,
            'domain_indices': domain_indices
        }
        
        # Cache if requested
        if cache_name and self.cache_dir:
            cache_path = self.cache_dir / f"{cache_name}.npz"
            print(f"Caching embeddings to {cache_path}")
            np.savez_compressed(cache_path, **result)
        
        print(f"✓ Extracted embeddings: {embeddings.shape}")
        
        return result
    
    def get_text_prompts(
        self,
        class_names: List[str] = ['real', 'fake'],
        domain_names: Optional[List[str]] = None,
        templates: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Generate text prompt embeddings for classification.
        
        Args:
            class_names: List of class names
            domain_names: Optional list of domain names
            templates: Optional custom templates
            
        Returns:
            Dictionary with prompt embeddings
        """
        if templates is None:
            templates = [
                "a photo of a {} image",
                "a {} photo",
                "{} content",
                "this is {} imagery"
            ]
        
        prompts = {}
        
        # Class prompts
        for class_name in class_names:
            class_prompts = [template.format(class_name) for template in templates]
            embeddings = self.encode_text(class_prompts, normalize=True)
            prompts[f'class_{class_name}'] = embeddings.mean(dim=0, keepdim=True)
        
        # Domain-specific prompts (if provided)
        if domain_names:
            for domain in domain_names:
                for class_name in class_names:
                    domain_class_prompts = [
                        f"a {class_name} image from {domain}",
                        f"{class_name} content generated by {domain}",
                        f"a photo of {class_name} {domain} imagery"
                    ]
                    embeddings = self.encode_text(domain_class_prompts, normalize=True)
                    prompts[f'{domain}_{class_name}'] = embeddings.mean(dim=0, keepdim=True)
        
        return prompts


def visualize_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    domain_indices: np.ndarray,
    domain_names: List[str],
    output_path: str,
    method: str = 'tsne'
):
    """
    Visualize CLIP embeddings using t-SNE or UMAP.
    
    Args:
        embeddings: CLIP embeddings [N, D]
        labels: Class labels [N]
        domain_indices: Domain indices [N]
        domain_names: List of domain names
        output_path: Path to save visualization
        method: 'tsne' or 'umap'
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print(f"Visualizing embeddings using {method.upper()}...")
    
    # Dimensionality reduction
    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    elif method == 'umap':
        from umap import UMAP
        reducer = UMAP(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Reduce dimensions
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Create visualizations
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot by domain
    ax = axes[0]
    for domain_idx in np.unique(domain_indices):
        mask = domain_indices == domain_idx
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            label=domain_names[domain_idx],
            alpha=0.6,
            s=20
        )
    ax.set_xlabel(f'{method.upper()} 1', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{method.upper()} 2', fontsize=12, fontweight='bold')
    ax.set_title('CLIP Embeddings by Domain', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)
    
    # Plot by class
    ax = axes[1]
    colors = ['#3498db', '#e74c3c']
    class_names = ['Real', 'Fake']
    for label_idx in [0, 1]:
        mask = labels == label_idx
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            label=class_names[label_idx],
            alpha=0.6,
            s=20,
            color=colors[label_idx]
        )
    ax.set_xlabel(f'{method.upper()} 1', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{method.upper()} 2', fontsize=12, fontweight='bold')
    ax.set_title('CLIP Embeddings by Class', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract CLIP embeddings for CDDB dataset')
    parser.add_argument('--data_root', type=str, default=r'd:\Farooq\genprompt\CDDB',
                       help='Path to CDDB dataset')
    parser.add_argument('--model', type=str, default='ViT-B/32',
                       choices=['ViT-B/32', 'ViT-L/14'],
                       help='CLIP model to use')
    parser.add_argument('--cache_dir', type=str, default='clip_embeddings',
                       help='Directory to cache embeddings')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for extraction')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'val'],
                       help='Dataset split')
    parser.add_argument('--visualize', action='store_true',
                       help='Create t-SNE visualization')
    
    args = parser.parse_args()
    
    # All domains except whichfaceisreal (if needed)
    all_domains = [
        'biggan', 'crn', 'cyclegan', 'deepfake', 'gaugan', 'glow',
        'imle', 'san', 'stargan_gf', 'stylegan', 'whichfaceisreal', 'wild'
    ]
    
    # Create dataset
    print(f"Creating dataset for {len(all_domains)} domains...")
    dataset = CDDBDataset(
        data_root=args.data_root,
        domains=all_domains,
        split=args.split,
        transform=get_clip_transforms(),
        return_domain_label=True
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Create CLIP encoder
    encoder = CLIPEncoder(
        model_name=args.model,
        cache_dir=args.cache_dir
    )
    
    # Extract embeddings
    cache_name = f"cddb_{args.split}_{args.model.replace('/', '_')}"
    result = encoder.extract_dataset_embeddings(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=4,
        cache_name=cache_name,
        force_recompute=False
    )
    
    # Print statistics
    print("\nEmbedding Statistics:")
    print(f"  Shape: {result['embeddings'].shape}")
    print(f"  Mean norm: {np.linalg.norm(result['embeddings'], axis=1).mean():.4f}")
    print(f"  Class distribution: Real={np.sum(result['labels']==0)}, Fake={np.sum(result['labels']==1)}")
    
    # Visualize if requested
    if args.visualize:
        output_path = Path(args.cache_dir) / f"{cache_name}_tsne.png"
        visualize_embeddings(
            embeddings=result['embeddings'],
            labels=result['labels'],
            domain_indices=result['domain_indices'],
            domain_names=all_domains,
            output_path=str(output_path),
            method='tsne'
        )
    
    print("\n✓ CLIP embedding extraction complete!")
