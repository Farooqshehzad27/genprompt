"""
Dataset Analysis Script for CDDB Dataset
Analyzes the structure and statistics of the CDDB deepfake dataset.
"""

import os
import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def analyze_domain_structure(domain_path):
    """
    Analyze the structure of a single domain.
    
    Returns:
        dict: Statistics including sample counts and subdomain information
    """
    domain_name = os.path.basename(domain_path)
    stats = {
        'domain': domain_name,
        'has_subdomains': False,
        'subdomains': [],
        'train_real': 0,
        'train_fake': 0,
        'val_real': 0,
        'val_fake': 0,
        'total': 0
    }
    
    # Check for train and val directories
    train_path = os.path.join(domain_path, 'train')
    val_path = os.path.join(domain_path, 'val')
    
    if not os.path.exists(train_path):
        print(f"Warning: No train directory found in {domain_name}")
        return stats
    
    # Check if domain has subdomain structure
    train_contents = os.listdir(train_path)
    
    # Check if domain has subdomain structure (cyclegan, glow, stargan_gf, stylegan)
    if domain_name in ['cyclegan', 'glow', 'stargan_gf', 'stylegan']:
        stats['has_subdomains'] = True
        
        # Get train subdomains
        train_subdomains = [d for d in os.listdir(train_path) 
                           if os.path.isdir(os.path.join(train_path, d)) 
                           and d not in ['0_real', '1_fake']]
        
        # Get val subdomains (may be different from train)
        val_subdomains = []
        if os.path.exists(val_path):
            val_subdomains = [d for d in os.listdir(val_path) 
                             if os.path.isdir(os.path.join(val_path, d)) 
                             and d not in ['0_real', '1_fake']]
        
        # Combine all unique subdomains for reporting
        all_subdomains = sorted(set(train_subdomains + val_subdomains))
        stats['subdomains'] = all_subdomains
        
        # Count train samples across all train subdomains
        for subdomain in train_subdomains:
            subdomain_train_path = os.path.join(train_path, subdomain)
            
            train_real_path = os.path.join(subdomain_train_path, '0_real')
            train_fake_path = os.path.join(subdomain_train_path, '1_fake')
            
            if os.path.exists(train_real_path):
                stats['train_real'] += len([f for f in os.listdir(train_real_path) 
                                           if os.path.isfile(os.path.join(train_real_path, f))])
            if os.path.exists(train_fake_path):
                stats['train_fake'] += len([f for f in os.listdir(train_fake_path) 
                                           if os.path.isfile(os.path.join(train_fake_path, f))])
        
        # Count val samples across all val subdomains (may differ from train)
        for subdomain in val_subdomains:
            subdomain_val_path = os.path.join(val_path, subdomain)
            
            val_real_path = os.path.join(subdomain_val_path, '0_real')
            val_fake_path = os.path.join(subdomain_val_path, '1_fake')
            
            if os.path.exists(val_real_path):
                stats['val_real'] += len([f for f in os.listdir(val_real_path) 
                                         if os.path.isfile(os.path.join(val_real_path, f))])
            if os.path.exists(val_fake_path):
                stats['val_fake'] += len([f for f in os.listdir(val_fake_path) 
                                         if os.path.isfile(os.path.join(val_fake_path, f))])
    else:
        # Standard structure without subdomains
        train_real_path = os.path.join(train_path, '0_real')
        train_fake_path = os.path.join(train_path, '1_fake')
        
        if os.path.exists(train_real_path):
            stats['train_real'] = len([f for f in os.listdir(train_real_path) 
                                      if os.path.isfile(os.path.join(train_real_path, f))])
        if os.path.exists(train_fake_path):
            stats['train_fake'] = len([f for f in os.listdir(train_fake_path) 
                                      if os.path.isfile(os.path.join(train_fake_path, f))])
        
        # Count val samples
        if os.path.exists(val_path):
            val_real_path = os.path.join(val_path, '0_real')
            val_fake_path = os.path.join(val_path, '1_fake')
            
            if os.path.exists(val_real_path):
                stats['val_real'] = len([f for f in os.listdir(val_real_path) 
                                        if os.path.isfile(os.path.join(val_real_path, f))])
            if os.path.exists(val_fake_path):
                stats['val_fake'] = len([f for f in os.listdir(val_fake_path) 
                                        if os.path.isfile(os.path.join(val_fake_path, f))])
    
    stats['total'] = stats['train_real'] + stats['train_fake'] + stats['val_real'] + stats['val_fake']
    
    return stats


def analyze_cddb_dataset(data_root):
    """
    Analyze the entire CDDB dataset.
    
    Args:
        data_root: Path to CDDB root directory
        
    Returns:
        list: List of domain statistics
    """
    data_root = Path(data_root)
    
    if not data_root.exists():
        raise ValueError(f"Data root {data_root} does not exist")
    
    # Get all domain directories
    domains = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    
    print(f"Found {len(domains)} domains in CDDB dataset:")
    print(", ".join(domains))
    print("\n" + "="*80 + "\n")
    
    all_stats = []
    
    for domain in domains:
        domain_path = os.path.join(data_root, domain)
        print(f"Analyzing domain: {domain}")
        
        stats = analyze_domain_structure(domain_path)
        all_stats.append(stats)
        
        # Print domain statistics
        print(f"  Has subdomains: {stats['has_subdomains']}")
        if stats['has_subdomains']:
            print(f"  Subdomains: {', '.join(stats['subdomains'])}")
        print(f"  Train - Real: {stats['train_real']:,}, Fake: {stats['train_fake']:,}")
        print(f"  Val   - Real: {stats['val_real']:,}, Fake: {stats['val_fake']:,}")
        print(f"  Total samples: {stats['total']:,}")
        print()
    
    return all_stats


def create_visualizations(stats, output_dir):
    """
    Create visualizations of dataset statistics.
    
    Args:
        stats: List of domain statistics
        output_dir: Directory to save visualizations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(stats)
    
    # 1. Total samples per domain
    plt.figure(figsize=(14, 6))
    plt.bar(df['domain'], df['total'], color='steelblue', alpha=0.8)
    plt.xlabel('Domain', fontsize=12, fontweight='bold')
    plt.ylabel('Total Samples', fontsize=12, fontweight='bold')
    plt.title('Total Samples per Domain in CDDB Dataset', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'total_samples_per_domain.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'total_samples_per_domain.png'}")
    plt.close()
    
    # 2. Train vs Val split
    fig, ax = plt.subplots(figsize=(14, 6))
    x = range(len(df))
    width = 0.35
    
    train_total = df['train_real'] + df['train_fake']
    val_total = df['val_real'] + df['val_fake']
    
    ax.bar([i - width/2 for i in x], train_total, width, label='Train', color='#2ecc71', alpha=0.8)
    ax.bar([i + width/2 for i in x], val_total, width, label='Val', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Domain', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax.set_title('Train vs Validation Split per Domain', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['domain'], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'train_val_split.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'train_val_split.png'}")
    plt.close()
    
    # 3. Real vs Fake distribution
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Train split
    axes[0].bar(df['domain'], df['train_real'], label='Real', color='#3498db', alpha=0.8)
    axes[0].bar(df['domain'], df['train_fake'], bottom=df['train_real'], label='Fake', color='#e67e22', alpha=0.8)
    axes[0].set_xlabel('Domain', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    axes[0].set_title('Train Set: Real vs Fake', fontsize=13, fontweight='bold')
    axes[0].set_xticklabels(df['domain'], rotation=45, ha='right')
    axes[0].legend(fontsize=11)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Val split
    axes[1].bar(df['domain'], df['val_real'], label='Real', color='#3498db', alpha=0.8)
    axes[1].bar(df['domain'], df['val_fake'], bottom=df['val_real'], label='Fake', color='#e67e22', alpha=0.8)
    axes[1].set_xlabel('Domain', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    axes[1].set_title('Validation Set: Real vs Fake', fontsize=13, fontweight='bold')
    axes[1].set_xticklabels(df['domain'], rotation=45, ha='right')
    axes[1].legend(fontsize=11)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'real_vs_fake_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'real_vs_fake_distribution.png'}")
    plt.close()
    
    # 4. Subdomain structure visualization
    subdomain_domains = df[df['has_subdomains'] == True]
    if len(subdomain_domains) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for idx, row in subdomain_domains.iterrows():
            ax.barh(row['domain'], len(row['subdomains']), color='#9b59b6', alpha=0.8)
        
        ax.set_xlabel('Number of Subdomains', fontsize=12, fontweight='bold')
        ax.set_ylabel('Domain', fontsize=12, fontweight='bold')
        ax.set_title('Domains with Subdomain Structure', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'subdomain_structure.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir / 'subdomain_structure.png'}")
        plt.close()
    
    # 5. Summary statistics table
    summary_df = df[['domain', 'train_real', 'train_fake', 'val_real', 'val_fake', 'total']].copy()
    summary_df['train_total'] = summary_df['train_real'] + summary_df['train_fake']
    summary_df['val_total'] = summary_df['val_real'] + summary_df['val_fake']
    
    # Add totals row
    totals = {
        'domain': 'TOTAL',
        'train_real': summary_df['train_real'].sum(),
        'train_fake': summary_df['train_fake'].sum(),
        'val_real': summary_df['val_real'].sum(),
        'val_fake': summary_df['val_fake'].sum(),
        'total': summary_df['total'].sum(),
        'train_total': summary_df['train_total'].sum(),
        'val_total': summary_df['val_total'].sum()
    }
    summary_df = pd.concat([summary_df, pd.DataFrame([totals])], ignore_index=True)
    
    # Create table visualization
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    headers = ['Domain', 'Train\nReal', 'Train\nFake', 'Train\nTotal', 'Val\nReal', 'Val\nFake', 'Val\nTotal', 'Total']
    
    for _, row in summary_df.iterrows():
        table_data.append([
            row['domain'],
            f"{int(row['train_real']):,}",
            f"{int(row['train_fake']):,}",
            f"{int(row['train_total']):,}",
            f"{int(row['val_real']):,}",
            f"{int(row['val_fake']):,}",
            f"{int(row['val_total']):,}",
            f"{int(row['total']):,}"
        ])
    
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center',
                     colWidths=[0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style total row
    for i in range(len(headers)):
        table[(len(table_data), i)].set_facecolor('#95a5a6')
        table[(len(table_data), i)].set_text_props(weight='bold')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
    
    plt.title('CDDB Dataset Statistics Summary', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'dataset_summary_table.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'dataset_summary_table.png'}")
    plt.close()


def save_statistics_json(stats, output_path):
    """Save statistics to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved statistics to: {output_path}")


def print_summary(stats):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("CDDB DATASET SUMMARY")
    print("="*80)
    
    total_samples = sum(s['total'] for s in stats)
    total_train = sum(s['train_real'] + s['train_fake'] for s in stats)
    total_val = sum(s['val_real'] + s['val_fake'] for s in stats)
    total_real = sum(s['train_real'] + s['val_real'] for s in stats)
    total_fake = sum(s['train_fake'] + s['val_fake'] for s in stats)
    
    subdomain_count = sum(1 for s in stats if s['has_subdomains'])
    
    print(f"\nTotal Domains: {len(stats)}")
    print(f"Domains with Subdomains: {subdomain_count}")
    print(f"  - {', '.join([s['domain'] for s in stats if s['has_subdomains']])}")
    print(f"\nTotal Samples: {total_samples:,}")
    print(f"  Train: {total_train:,} ({total_train/total_samples*100:.1f}%)")
    print(f"  Val:   {total_val:,} ({total_val/total_samples*100:.1f}%)")
    print(f"\nClass Distribution:")
    print(f"  Real: {total_real:,} ({total_real/total_samples*100:.1f}%)")
    print(f"  Fake: {total_fake:,} ({total_fake/total_samples*100:.1f}%)")
    print("\n" + "="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze CDDB dataset structure and statistics')
    parser.add_argument('--data_root', type=str, default=r'd:\Farooq\genprompt\CDDB',
                       help='Path to CDDB dataset root directory')
    parser.add_argument('--output_dir', type=str, default='dataset_analysis',
                       help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    print("Starting CDDB Dataset Analysis...")
    print(f"Data root: {args.data_root}")
    print(f"Output directory: {args.output_dir}\n")
    
    # Analyze dataset
    stats = analyze_cddb_dataset(args.data_root)
    
    # Print summary
    print_summary(stats)
    
    # Save JSON statistics
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_statistics_json(stats, output_dir / 'dataset_statistics.json')
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualizations(stats, output_dir)
    
    print(f"\n✓ Analysis complete! Results saved to: {output_dir}")
