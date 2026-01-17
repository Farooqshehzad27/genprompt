"""
Quick Analysis Script
Analyze training results and generate summary report.
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_training_results(experiment_dir: str):
    """
    Analyze training results from experiment directory.
    
    Args:
        experiment_dir: Path to experiment directory with checkpoints
    """
    experiment_dir = Path(experiment_dir)
    checkpoint_dir = experiment_dir / 'checkpoints'
    
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        return
    
    # Get all checkpoints
    checkpoints = sorted(checkpoint_dir.glob('checkpoint_*.pt'))
    
    if not checkpoints:
        print(f"Error: No checkpoints found in {checkpoint_dir}")
        return
    
    print(f"Found {len(checkpoints)} checkpoints")
    print(f"{'='*80}")
    
    # Analyze each checkpoint
    results = []
    
    for ckpt_path in checkpoints:
        import torch
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        
        domain_name = checkpoint['domain_name']
        seen_domains = checkpoint['seen_domains']
        cluster_to_domain = checkpoint['cluster_to_domain']
        
        # Count prompts
        prompt_pool_state = checkpoint['prompt_pool']
        n_prompts = len([k for k in prompt_pool_state.keys() if 'visual_prompts' in k])
        
        results.append({
            'domain': domain_name,
            'n_seen_domains': len(seen_domains),
            'n_prompts': n_prompts,
            'n_clusters': len(cluster_to_domain)
        })
        
        print(f"\nAfter training on: {domain_name}")
        print(f"  Seen domains: {', '.join(seen_domains)}")
        print(f"  Number of prompts: {n_prompts}")
        print(f"  Number of clusters: {len(cluster_to_domain)}")
    
    print(f"\n{'='*80}")
    print("Training Summary")
    print(f"{'='*80}")
    print(f"Total domains trained: {len(results)}")
    print(f"Final number of prompts: {results[-1]['n_prompts']}")
    print(f"Final number of clusters: {results[-1]['n_clusters']}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Prompts over time
    ax = axes[0]
    domains = [r['domain'] for r in results]
    n_prompts = [r['n_prompts'] for r in results]
    
    ax.plot(range(len(domains)), n_prompts, marker='o', linewidth=2, markersize=8, color='steelblue')
    ax.set_xlabel('Training Step (Domain)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Prompts', fontsize=12, fontweight='bold')
    ax.set_title('Prompt Allocation Over Time', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(domains, rotation=45, ha='right')
    ax.grid(alpha=0.3)
    
    # Plot 2: Clusters over time
    ax = axes[1]
    n_clusters = [r['n_clusters'] for r in results]
    
    ax.plot(range(len(domains)), n_clusters, marker='s', linewidth=2, markersize=8, color='teal')
    ax.set_xlabel('Training Step (Domain)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Clusters', fontsize=12, fontweight='bold')
    ax.set_title('Cluster Discovery Over Time', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(domains, rotation=45, ha='right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_path = experiment_dir / 'training_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved training analysis to: {output_path}")
    plt.close()
    
    # Save results to JSON
    output_json = experiment_dir / 'training_summary.json'
    with open(output_json, 'w') as f:
        json.dump({
            'checkpoints': results,
            'summary': {
                'total_domains': len(results),
                'final_prompts': results[-1]['n_prompts'],
                'final_clusters': results[-1]['n_clusters']
            }
        }, f, indent=2)
    
    print(f"✓ Saved training summary to: {output_json}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze training results')
    parser.add_argument('--experiment_dir', type=str, required=True,
                       help='Path to experiment directory')
    
    args = parser.parse_args()
    
    analyze_training_results(args.experiment_dir)
