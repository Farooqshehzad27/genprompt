"""
Evaluation Script for GenPrompt
Comprehensive evaluation including forgetting, generalization, and cluster quality.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Import our modules
from data_loader import CDDBDataset, get_clip_transforms
from clip_encoder import CLIPEncoder
from domain_discovery import ClusterEvaluator, visualize_clusters
from prompt_pool import PromptPool
from prompt_router import PromptRoutingNetwork
from contrastive_loss import PrototypeContrastiveLearning


class GenPromptEvaluator:
    """
    Comprehensive evaluation for GenPrompt.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        data_root: str,
        clip_model: str = 'ViT-B/32',
        device: str = 'cuda',
        cache_dir: str = 'clip_embeddings'
    ):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint
            data_root: Path to CDDB dataset
            clip_model: CLIP model name
            device: Device to run on
            cache_dir: Directory for cached embeddings
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.data_root = data_root
        self.clip_model = clip_model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.cache_dir = Path(cache_dir)
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.seen_domains = checkpoint['seen_domains']
        self.cluster_to_domain = checkpoint['cluster_to_domain']
        
        print(f"Checkpoint trained on {len(self.seen_domains)} domains:")
        print(f"  {', '.join(self.seen_domains)}")
        
        # Initialize CLIP encoder
        self.clip_encoder = CLIPEncoder(
            model_name=clip_model,
            device=str(self.device),
            cache_dir=str(self.cache_dir)
        )
        embed_dim = self.clip_encoder.embed_dim
        
        # Load prompt pool
        self.prompt_pool = PromptPool(
            visual_prompt_length=10,
            textual_n_ctx=4,
            embed_dim=embed_dim,
            max_prompts=50
        ).to(self.device)
        # Load prompt pool (use strict=False to handle checkpoint format)
        self.prompt_pool.load_state_dict(checkpoint['prompt_pool'], strict=False)
        
        # Manually reconstruct prompt pool from checkpoint if needed
        if self.prompt_pool.n_prompts == 0:
            print("Reconstructing prompt pool from checkpoint...")
            # Count prompts in checkpoint
            visual_prompt_keys = [k for k in checkpoint['prompt_pool'].keys() if 'visual_prompts' in k and 'prompt_tokens' in k]
            n_prompts_in_ckpt = len(set([k.split('.')[1] for k in visual_prompt_keys]))
            
            # Allocate prompts
            for i in range(n_prompts_in_ckpt):
                self.prompt_pool.allocate_prompt(cluster_id=i)
            
            # Load state dict again
            self.prompt_pool.load_state_dict(checkpoint['prompt_pool'], strict=False)
        
        self.prompt_pool.eval()
        
        # Load PRN
        self.prn = PromptRoutingNetwork(
            embed_dim=embed_dim,
            hidden_dim=256,
            n_prompts=self.prompt_pool.n_prompts,
            n_classes=2,
            routing_mode='soft'
        ).to(self.device)
        self.prn.load_state_dict(checkpoint['prn'])
        self.prn.eval()
        
        print(f"✓ Model loaded successfully")
        print(f"  Number of prompts: {self.prompt_pool.n_prompts}")
        print(f"  Embed dim: {embed_dim}")
    
    @torch.no_grad()
    def evaluate_domain(
        self,
        domain_name: str,
        split: str = 'val',
        batch_size: int = 64
    ) -> Dict:
        """
        Evaluate on a single domain.
        
        Args:
            domain_name: Domain to evaluate
            split: 'train' or 'val'
            batch_size: Batch size
            
        Returns:
            Dictionary with metrics
        """
        print(f"\nEvaluating on {domain_name} ({split})...")
        
        # Create dataset
        dataset = CDDBDataset(
            data_root=self.data_root,
            domains=[domain_name],
            split=split,
            transform=get_clip_transforms(),
            return_domain_label=False
        )
        
        if len(dataset) == 0:
            print(f"  Warning: No samples found for {domain_name} {split}")
            return None
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        all_preds = []
        all_labels = []
        all_probs = []
        all_routing_weights = []
        
        for images, labels in tqdm(dataloader, desc=f"  {domain_name}"):
            images = images.to(self.device)
            
            # Get CLIP embeddings
            embeddings = self.clip_encoder.encode_images(images, normalize=True)
            
            # Convert to float32 if needed
            if embeddings.dtype != torch.float32:
                embeddings = embeddings.float()
            
            # Forward pass
            logits, routing_weights, expert_logits = self.prn(embeddings)
            
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_routing_weights.extend(routing_weights.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_routing_weights = np.array(all_routing_weights)
        
        # Compute metrics
        accuracy = 100 * (all_preds == all_labels).mean()
        
        # AUC (if both classes present)
        if len(np.unique(all_labels)) == 2:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            auc = None
        
        # Per-class accuracy
        real_mask = all_labels == 0
        fake_mask = all_labels == 1
        real_acc = 100 * (all_preds[real_mask] == all_labels[real_mask]).mean() if real_mask.sum() > 0 else 0
        fake_acc = 100 * (all_preds[fake_mask] == all_labels[fake_mask]).mean() if fake_mask.sum() > 0 else 0
        
        # Routing statistics
        routing_entropy = -(all_routing_weights * np.log(all_routing_weights + 1e-10)).sum(axis=1).mean()
        routing_concentration = all_routing_weights.max(axis=1).mean()
        
        results = {
            'domain': domain_name,
            'split': split,
            'n_samples': len(dataset),
            'accuracy': accuracy,
            'auc': auc,
            'real_accuracy': real_acc,
            'fake_accuracy': fake_acc,
            'routing_entropy': routing_entropy,
            'routing_concentration': routing_concentration,
            'predictions': all_preds,
            'labels': all_labels,
            'routing_weights': all_routing_weights
        }
        
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  AUC: {auc:.4f}" if auc else "  AUC: N/A")
        print(f"  Real Acc: {real_acc:.2f}%, Fake Acc: {fake_acc:.2f}%")
        
        return results
    
    def evaluate_all_domains(
        self,
        domains: List[str],
        split: str = 'val'
    ) -> Dict:
        """Evaluate on all specified domains."""
        all_results = {}
        
        for domain in domains:
            results = self.evaluate_domain(domain, split=split)
            if results:
                all_results[domain] = results
        
        # Compute average metrics
        avg_accuracy = np.mean([r['accuracy'] for r in all_results.values()])
        avg_auc = np.mean([r['auc'] for r in all_results.values() if r['auc'] is not None])
        
        print(f"\n{'='*80}")
        print(f"Overall Results ({split}):")
        print(f"  Average Accuracy: {avg_accuracy:.2f}%")
        print(f"  Average AUC: {avg_auc:.4f}")
        print(f"{'='*80}")
        
        return {
            'per_domain': all_results,
            'average_accuracy': avg_accuracy,
            'average_auc': avg_auc
        }
    
    def measure_forgetting(
        self,
        checkpoint_dir: str,
        domains: List[str]
    ) -> Dict:
        """
        Measure catastrophic forgetting by comparing performance
        after training on each domain.
        
        Args:
            checkpoint_dir: Directory containing checkpoints
            domains: List of domains in training order
            
        Returns:
            Forgetting metrics
        """
        print(f"\n{'='*80}")
        print("Measuring Catastrophic Forgetting")
        print(f"{'='*80}")
        
        checkpoint_dir = Path(checkpoint_dir)
        
        # Store accuracy after each domain
        accuracy_matrix = np.zeros((len(domains), len(domains)))
        accuracy_matrix[:] = np.nan
        
        for i, current_domain in enumerate(domains):
            checkpoint_path = checkpoint_dir / f'checkpoint_{current_domain}.pt'
            
            if not checkpoint_path.exists():
                print(f"Warning: Checkpoint not found for {current_domain}")
                continue
            
            # Load checkpoint
            evaluator = GenPromptEvaluator(
                checkpoint_path=str(checkpoint_path),
                data_root=self.data_root,
                clip_model=self.clip_model,
                device=str(self.device),
                cache_dir=str(self.cache_dir)
            )
            
            # Evaluate on all domains seen so far
            for j, eval_domain in enumerate(domains[:i+1]):
                result = evaluator.evaluate_domain(eval_domain, split='val', batch_size=64)
                if result:
                    accuracy_matrix[i, j] = result['accuracy']
        
        # Compute forgetting metrics
        forgetting_scores = []
        
        for j in range(len(domains) - 1):
            # Accuracy on domain j after training on domain j
            initial_acc = accuracy_matrix[j, j]
            # Accuracy on domain j after training on all domains
            final_acc = accuracy_matrix[-1, j]
            
            if not np.isnan(initial_acc) and not np.isnan(final_acc):
                forgetting = initial_acc - final_acc
                forgetting_scores.append(forgetting)
        
        avg_forgetting = np.mean(forgetting_scores) if forgetting_scores else 0
        
        print(f"\nForgetting Analysis:")
        print(f"  Average Forgetting: {avg_forgetting:.2f}%")
        print(f"  (Negative = improvement, Positive = forgetting)")
        
        return {
            'accuracy_matrix': accuracy_matrix,
            'forgetting_scores': forgetting_scores,
            'average_forgetting': avg_forgetting
        }
    
    def create_visualizations(
        self,
        results: Dict,
        output_dir: str
    ):
        """Create comprehensive visualizations."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        per_domain = results['per_domain']
        
        # 1. Accuracy bar chart
        plt.figure(figsize=(12, 6))
        domains = list(per_domain.keys())
        accuracies = [per_domain[d]['accuracy'] for d in domains]
        
        plt.bar(domains, accuracies, color='steelblue', alpha=0.8)
        plt.axhline(y=results['average_accuracy'], color='r', linestyle='--', 
                   label=f"Average: {results['average_accuracy']:.2f}%")
        plt.xlabel('Domain', fontsize=12, fontweight='bold')
        plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        plt.title('Per-Domain Accuracy', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'accuracy_per_domain.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {output_dir / 'accuracy_per_domain.png'}")
        
        # 2. Confusion matrices
        for domain, result in per_domain.items():
            cm = confusion_matrix(result['labels'], result['predictions'])
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Real', 'Fake'],
                       yticklabels=['Real', 'Fake'])
            plt.title(f'Confusion Matrix: {domain}', fontsize=14, fontweight='bold')
            plt.ylabel('True Label', fontsize=12)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.tight_layout()
            plt.savefig(output_dir / f'confusion_matrix_{domain}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Saved confusion matrices to {output_dir}")
        
        # 3. Routing weight visualization
        plt.figure(figsize=(14, 8))
        
        for idx, (domain, result) in enumerate(per_domain.items()):
            routing_weights = result['routing_weights']
            avg_routing = routing_weights.mean(axis=0)
            
            plt.subplot(2, 5, idx + 1)
            plt.bar(range(len(avg_routing)), avg_routing, color='teal', alpha=0.7)
            plt.title(domain, fontsize=10, fontweight='bold')
            plt.xlabel('Prompt ID', fontsize=9)
            plt.ylabel('Avg Weight', fontsize=9)
            plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'routing_weights.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {output_dir / 'routing_weights.png'}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate GenPrompt model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, default=r'd:\Farooq\genprompt\CDDB',
                       help='Path to CDDB dataset')
    parser.add_argument('--clip_model', type=str, default='ViT-B/32',
                       help='CLIP model used for training')
    parser.add_argument('--domains', type=str, nargs='+',
                       default=['biggan', 'crn', 'cyclegan', 'deepfake', 'gaugan', 
                               'glow', 'imle', 'san', 'stargan_gf', 'stylegan'],
                       help='Domains to evaluate')
    parser.add_argument('--test_domains', type=str, nargs='+',
                       default=['whichfaceisreal', 'wild'],
                       help='Held-out domains for generalization test')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val'],
                       help='Dataset split to evaluate')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--measure_forgetting', action='store_true',
                       help='Measure catastrophic forgetting')
    parser.add_argument('--checkpoint_dir', type=str,
                       help='Directory with all checkpoints (for forgetting analysis)')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = GenPromptEvaluator(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        clip_model=args.clip_model
    )
    
    # Evaluate on training domains
    print(f"\n{'='*80}")
    print("Evaluating on Training Domains")
    print(f"{'='*80}")
    
    results = evaluator.evaluate_all_domains(args.domains, split=args.split)
    
    # Evaluate on held-out domains (generalization)
    if args.test_domains:
        print(f"\n{'='*80}")
        print("Evaluating on Held-Out Domains (Generalization)")
        print(f"{'='*80}")
        
        test_results = evaluator.evaluate_all_domains(args.test_domains, split=args.split)
        results['generalization'] = test_results
    
    # Measure forgetting
    if args.measure_forgetting and args.checkpoint_dir:
        forgetting_results = evaluator.measure_forgetting(
            checkpoint_dir=args.checkpoint_dir,
            domains=args.domains
        )
        results['forgetting'] = forgetting_results
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            """Convert numpy types to JSON-serializable types."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        results_json = convert_to_serializable(results)
        json.dump(results_json, f, indent=2)
    
    print(f"\n✓ Results saved to {output_dir / 'evaluation_results.json'}")
    print(f"\n✓ Evaluation complete!")
    print(f"\nTo generate visualizations, run:")
    print(f"  python visualize_results.py --results {output_dir / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()
