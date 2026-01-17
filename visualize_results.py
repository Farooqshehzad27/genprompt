"""
Visualization Script for GenPrompt Evaluation Results
Standalone script to create visualizations from evaluation_results.json
"""

import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def load_results(results_path: str) -> dict:
    """Load evaluation results from JSON file."""
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results


def plot_accuracy_comparison(results: dict, output_dir: Path):
    """
    Plot accuracy comparison between training and held-out domains.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Training domains
    ax = axes[0]
    per_domain = results['per_domain']
    domains = list(per_domain.keys())
    accuracies = [per_domain[d]['accuracy'] for d in domains]
    
    colors = ['steelblue'] * len(domains)
    bars = ax.bar(domains, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add average line
    avg_acc = results['average_accuracy']
    ax.axhline(y=avg_acc, color='red', linestyle='--', linewidth=2,
               label=f'Average: {avg_acc:.2f}%')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Training Domain', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Performance on Training Domains', fontsize=15, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.set_xticklabels(domains, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Held-out domains (if available)
    if 'generalization' in results:
        ax = axes[1]
        gen_results = results['generalization']['per_domain']
        test_domains = list(gen_results.keys())
        test_accuracies = [gen_results[d]['accuracy'] for d in test_domains]
        
        colors = ['coral'] * len(test_domains)
        bars = ax.bar(test_domains, test_accuracies, color=colors, alpha=0.8, 
                     edgecolor='black', linewidth=1.5)
        
        # Add average line
        avg_test_acc = results['generalization']['average_accuracy']
        ax.axhline(y=avg_test_acc, color='darkred', linestyle='--', linewidth=2,
                   label=f'Average: {avg_test_acc:.2f}%')
        
        # Add training average for comparison
        ax.axhline(y=avg_acc, color='steelblue', linestyle=':', linewidth=2,
                   label=f'Training Avg: {avg_acc:.2f}%', alpha=0.7)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Held-Out Domain (Zero-Shot)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
        ax.set_title('Generalization to Unseen Domains', fontsize=15, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.set_xticklabels(test_domains, rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = output_dir / 'accuracy_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_confusion_matrices(results: dict, output_dir: Path, domain_type: str = 'training'):
    """
    Create confusion matrices for all domains.
    
    Args:
        domain_type: 'training' or 'generalization'
    """
    if domain_type == 'training':
        per_domain = results['per_domain']
        prefix = 'train'
    else:
        if 'generalization' not in results:
            print(f"No generalization results found, skipping...")
            return
        per_domain = results['generalization']['per_domain']
        prefix = 'test'
    
    for domain, result in per_domain.items():
        labels = np.array(result['labels'])
        predictions = np.array(result['predictions'])
        
        cm = confusion_matrix(labels, predictions)
        
        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Raw counts
        ax = axes[0]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Real', 'Fake'],
                   yticklabels=['Real', 'Fake'],
                   cbar_kws={'label': 'Count'},
                   ax=ax, annot_kws={'size': 14, 'weight': 'bold'})
        ax.set_title(f'{domain} - Raw Counts', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        
        # Normalized percentages
        ax = axes[1]
        sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Oranges',
                   xticklabels=['Real', 'Fake'],
                   yticklabels=['Real', 'Fake'],
                   cbar_kws={'label': 'Percentage (%)'},
                   ax=ax, annot_kws={'size': 14, 'weight': 'bold'})
        ax.set_title(f'{domain} - Normalized (%)', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        output_path = output_dir / f'confusion_matrix_{prefix}_{domain}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✓ Saved confusion matrices for {len(per_domain)} {domain_type} domains")


def plot_routing_weights(results: dict, output_dir: Path):
    """Visualize routing weight distributions."""
    per_domain = results['per_domain']
    
    n_domains = len(per_domain)
    n_cols = 5
    n_rows = (n_domains + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    axes = axes.flatten() if n_domains > 1 else [axes]
    
    for idx, (domain, result) in enumerate(per_domain.items()):
        ax = axes[idx]
        
        routing_weights = np.array(result['routing_weights'])
        avg_routing = routing_weights.mean(axis=0)
        std_routing = routing_weights.std(axis=0)
        
        x = np.arange(len(avg_routing))
        bars = ax.bar(x, avg_routing, color='teal', alpha=0.7, edgecolor='black')
        ax.errorbar(x, avg_routing, yerr=std_routing, fmt='none', 
                   ecolor='black', capsize=3, alpha=0.5)
        
        # Highlight top-3 prompts
        top_3_idx = np.argsort(avg_routing)[-3:]
        for i in top_3_idx:
            bars[i].set_color('darkgreen')
            bars[i].set_alpha(0.9)
        
        ax.set_title(f'{domain}\n(Top prompts: {", ".join(map(str, top_3_idx))})', 
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Prompt ID', fontsize=10)
        ax.set_ylabel('Avg Weight', fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_domains, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = output_dir / 'routing_weights_detailed.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_metrics_table(results: dict, output_dir: Path):
    """Create a comprehensive metrics table."""
    per_domain = results['per_domain']
    
    # Prepare data
    domains = []
    accuracies = []
    aucs = []
    real_accs = []
    fake_accs = []
    
    for domain, result in per_domain.items():
        domains.append(domain)
        accuracies.append(result['accuracy'])
        aucs.append(result['auc'] if result['auc'] is not None else 0)
        real_accs.append(result['real_accuracy'])
        fake_accs.append(result['fake_accuracy'])
    
    # Add generalization results if available
    if 'generalization' in results:
        gen_results = results['generalization']['per_domain']
        for domain, result in gen_results.items():
            domains.append(f"{domain}*")  # Mark as held-out
            accuracies.append(result['accuracy'])
            aucs.append(result['auc'] if result['auc'] is not None else 0)
            real_accs.append(result['real_accuracy'])
            fake_accs.append(result['fake_accuracy'])
    
    # Create table
    fig, ax = plt.subplots(figsize=(12, len(domains) * 0.5 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    for i in range(len(domains)):
        table_data.append([
            domains[i],
            f"{accuracies[i]:.2f}%",
            f"{aucs[i]:.4f}" if aucs[i] > 0 else "N/A",
            f"{real_accs[i]:.2f}%",
            f"{fake_accs[i]:.2f}%"
        ])
    
    # Add average rows
    train_avg_idx = len(per_domain)
    table_data.append(['─' * 15] * 5)
    table_data.append([
        'Training Avg',
        f"{results['average_accuracy']:.2f}%",
        f"{results['average_auc']:.4f}",
        f"{np.mean(real_accs[:train_avg_idx]):.2f}%",
        f"{np.mean(fake_accs[:train_avg_idx]):.2f}%"
    ])
    
    if 'generalization' in results:
        table_data.append([
            'Held-Out Avg*',
            f"{results['generalization']['average_accuracy']:.2f}%",
            f"{results['generalization']['average_auc']:.4f}",
            f"{np.mean(real_accs[train_avg_idx:]):.2f}%",
            f"{np.mean(fake_accs[train_avg_idx:]):.2f}%"
        ])
    
    table = ax.table(cellText=table_data,
                    colLabels=['Domain', 'Accuracy', 'AUC', 'Real Acc', 'Fake Acc'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.25, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style average rows
    for i in range(5):
        table[(len(table_data) - 2, i)].set_facecolor('#E7E6E6')
        table[(len(table_data) - 1, i)].set_facecolor('#FFF2CC')
        table[(len(table_data) - 1, i)].set_text_props(weight='bold')
    
    plt.title('GenPrompt Evaluation Results\n(* = Held-Out/Zero-Shot)', 
             fontsize=14, fontweight='bold', pad=20)
    
    output_path = output_dir / 'metrics_table.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize GenPrompt evaluation results')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to evaluation_results.json')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: same as results file)')
    parser.add_argument('--plots', type=str, nargs='+',
                       default=['all'],
                       choices=['all', 'accuracy', 'confusion', 'routing', 'table'],
                       help='Which plots to generate')
    
    args = parser.parse_args()
    
    # Load results
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        return
    
    results = load_results(results_path)
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_path.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating visualizations...")
    print(f"Results: {results_path}")
    print(f"Output: {output_dir}")
    print("=" * 80)
    
    # Generate plots
    plots_to_generate = args.plots
    if 'all' in plots_to_generate:
        plots_to_generate = ['accuracy', 'confusion', 'routing', 'table']
    
    if 'accuracy' in plots_to_generate:
        plot_accuracy_comparison(results, output_dir)
    
    if 'confusion' in plots_to_generate:
        plot_confusion_matrices(results, output_dir, domain_type='training')
        plot_confusion_matrices(results, output_dir, domain_type='generalization')
    
    if 'routing' in plots_to_generate:
        plot_routing_weights(results, output_dir)
    
    if 'table' in plots_to_generate:
        plot_metrics_table(results, output_dir)
    
    print("=" * 80)
    print(f"✓ Visualization complete! All plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
