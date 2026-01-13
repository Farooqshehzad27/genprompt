"""
Latent Deepfake Domain Discovery (L3D)
Unsupervised clustering in CLIP embedding space for domain discovery.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score,
    normalized_mutual_info_score, adjusted_rand_score
)


class DomainDiscovery:
    """
    Unsupervised domain discovery using clustering in CLIP space.
    """
    
    def __init__(
        self,
        method: str = 'kmeans',
        n_clusters: Optional[int] = None,
        random_state: int = 42,
        **kwargs
    ):
        """
        Args:
            method: Clustering method ('kmeans', 'dbscan', 'agglomerative')
            n_clusters: Number of clusters (for kmeans/agglomerative)
            random_state: Random seed
            **kwargs: Additional arguments for clustering algorithm
        """
        self.method = method
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kwargs = kwargs
        
        self.clusterer = None
        self.cluster_assignments = None
        self.cluster_prototypes = None
        self.n_discovered_clusters = 0
        
    def fit(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fit clustering model and discover domains.
        
        Args:
            embeddings: CLIP embeddings [N, D]
            
        Returns:
            cluster_assignments: Cluster labels [N]
        """
        print(f"Discovering domains using {self.method}...")
        
        if self.method == 'kmeans':
            if self.n_clusters is None:
                raise ValueError("n_clusters must be specified for kmeans")
            
            self.clusterer = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                **self.kwargs
            )
            self.cluster_assignments = self.clusterer.fit_predict(embeddings)
            
        elif self.method == 'dbscan':
            eps = self.kwargs.get('eps', 0.5)
            min_samples = self.kwargs.get('min_samples', 10)
            
            self.clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            self.cluster_assignments = self.clusterer.fit_predict(embeddings)
            
        elif self.method == 'agglomerative':
            if self.n_clusters is None:
                raise ValueError("n_clusters must be specified for agglomerative")
            
            self.clusterer = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                **self.kwargs
            )
            self.cluster_assignments = self.clusterer.fit_predict(embeddings)
            
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")
        
        # Compute cluster prototypes
        self._compute_prototypes(embeddings)
        
        # Count discovered clusters (excluding noise for DBSCAN)
        unique_clusters = np.unique(self.cluster_assignments)
        self.n_discovered_clusters = len(unique_clusters[unique_clusters >= 0])
        
        print(f"Discovered {self.n_discovered_clusters} domains")
        
        return self.cluster_assignments
    
    def _compute_prototypes(self, embeddings: np.ndarray):
        """Compute cluster prototypes as mean of embeddings."""
        unique_clusters = np.unique(self.cluster_assignments)
        unique_clusters = unique_clusters[unique_clusters >= 0]  # Exclude noise (-1)
        
        self.cluster_prototypes = {}
        for cluster_id in unique_clusters:
            mask = self.cluster_assignments == cluster_id
            prototype = embeddings[mask].mean(axis=0)
            # L2 normalize
            prototype = prototype / np.linalg.norm(prototype)
            self.cluster_prototypes[int(cluster_id)] = prototype
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Assign new embeddings to discovered clusters.
        
        Args:
            embeddings: CLIP embeddings [N, D]
            
        Returns:
            cluster_assignments: Cluster labels [N]
        """
        if self.cluster_prototypes is None:
            raise ValueError("Must call fit() before predict()")
        
        # Assign to nearest prototype
        assignments = []
        for emb in embeddings:
            emb = emb / np.linalg.norm(emb)  # L2 normalize
            
            max_sim = -1
            best_cluster = 0
            
            for cluster_id, prototype in self.cluster_prototypes.items():
                sim = np.dot(emb, prototype)
                if sim > max_sim:
                    max_sim = sim
                    best_cluster = cluster_id
            
            assignments.append(best_cluster)
        
        return np.array(assignments)
    
    def update_prototypes(
        self,
        embeddings: np.ndarray,
        cluster_ids: np.ndarray,
        momentum: float = 0.9
    ):
        """
        Update cluster prototypes with new data (for online learning).
        
        Args:
            embeddings: New CLIP embeddings [N, D]
            cluster_ids: Cluster assignments [N]
            momentum: Momentum for exponential moving average
        """
        for cluster_id in np.unique(cluster_ids):
            if cluster_id < 0:  # Skip noise
                continue
            
            mask = cluster_ids == cluster_id
            new_prototype = embeddings[mask].mean(axis=0)
            new_prototype = new_prototype / np.linalg.norm(new_prototype)
            
            if cluster_id in self.cluster_prototypes:
                # EMA update
                old_prototype = self.cluster_prototypes[cluster_id]
                updated_prototype = momentum * old_prototype + (1 - momentum) * new_prototype
                updated_prototype = updated_prototype / np.linalg.norm(updated_prototype)
                self.cluster_prototypes[cluster_id] = updated_prototype
            else:
                # New cluster
                self.cluster_prototypes[cluster_id] = new_prototype
                self.n_discovered_clusters += 1


class ClusterEvaluator:
    """
    Evaluate clustering quality with various metrics.
    """
    
    @staticmethod
    def evaluate(
        embeddings: np.ndarray,
        cluster_assignments: np.ndarray,
        ground_truth_labels: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute clustering quality metrics.
        
        Args:
            embeddings: CLIP embeddings [N, D]
            cluster_assignments: Predicted cluster labels [N]
            ground_truth_labels: Optional ground truth domain labels [N]
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Filter out noise points (label -1 from DBSCAN)
        valid_mask = cluster_assignments >= 0
        valid_embeddings = embeddings[valid_mask]
        valid_assignments = cluster_assignments[valid_mask]
        
        if len(valid_assignments) == 0:
            print("Warning: No valid clusters found")
            return metrics
        
        # Intrinsic metrics (no ground truth needed)
        if len(np.unique(valid_assignments)) > 1:
            try:
                metrics['silhouette_score'] = silhouette_score(
                    valid_embeddings, valid_assignments
                )
            except:
                metrics['silhouette_score'] = -1.0
            
            try:
                metrics['davies_bouldin_score'] = davies_bouldin_score(
                    valid_embeddings, valid_assignments
                )
            except:
                metrics['davies_bouldin_score'] = -1.0
        
        metrics['n_clusters'] = len(np.unique(valid_assignments))
        metrics['n_noise_points'] = np.sum(cluster_assignments == -1)
        
        # Extrinsic metrics (require ground truth)
        if ground_truth_labels is not None:
            valid_gt = ground_truth_labels[valid_mask]
            
            metrics['nmi'] = normalized_mutual_info_score(
                valid_gt, valid_assignments
            )
            metrics['ari'] = adjusted_rand_score(
                valid_gt, valid_assignments
            )
            metrics['purity'] = ClusterEvaluator._compute_purity(
                valid_gt, valid_assignments
            )
        
        return metrics
    
    @staticmethod
    def _compute_purity(ground_truth: np.ndarray, cluster_assignments: np.ndarray) -> float:
        """Compute cluster purity."""
        total = len(ground_truth)
        purity_sum = 0
        
        for cluster_id in np.unique(cluster_assignments):
            mask = cluster_assignments == cluster_id
            cluster_gt = ground_truth[mask]
            
            if len(cluster_gt) > 0:
                # Most common ground truth label in this cluster
                most_common_count = np.bincount(cluster_gt).max()
                purity_sum += most_common_count
        
        return purity_sum / total
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float]):
        """Print metrics in a formatted way."""
        print("\nClustering Quality Metrics:")
        print("=" * 50)
        
        if 'n_clusters' in metrics:
            print(f"Number of clusters: {metrics['n_clusters']}")
        if 'n_noise_points' in metrics:
            print(f"Noise points: {metrics['n_noise_points']}")
        
        print("\nIntrinsic Metrics:")
        if 'silhouette_score' in metrics:
            print(f"  Silhouette Score: {metrics['silhouette_score']:.4f} (higher is better)")
        if 'davies_bouldin_score' in metrics:
            print(f"  Davies-Bouldin Score: {metrics['davies_bouldin_score']:.4f} (lower is better)")
        
        if any(k in metrics for k in ['nmi', 'ari', 'purity']):
            print("\nExtrinsic Metrics (vs Ground Truth):")
            if 'nmi' in metrics:
                print(f"  NMI: {metrics['nmi']:.4f}")
            if 'ari' in metrics:
                print(f"  ARI: {metrics['ari']:.4f}")
            if 'purity' in metrics:
                print(f"  Purity: {metrics['purity']:.4f}")
        
        print("=" * 50)


def find_optimal_k(
    embeddings: np.ndarray,
    k_range: range = range(2, 21),
    method: str = 'silhouette'
) -> Tuple[int, Dict]:
    """
    Find optimal number of clusters using elbow method or silhouette analysis.
    
    Args:
        embeddings: CLIP embeddings [N, D]
        k_range: Range of K values to try
        method: 'silhouette' or 'elbow'
        
    Returns:
        optimal_k: Best K value
        scores: Dictionary of K -> score
    """
    print(f"Finding optimal K using {method} method...")
    
    scores = {}
    
    for k in k_range:
        clusterer = KMeans(n_clusters=k, random_state=42)
        labels = clusterer.fit_predict(embeddings)
        
        if method == 'silhouette':
            score = silhouette_score(embeddings, labels)
        elif method == 'elbow':
            score = -clusterer.inertia_  # Negative because we want to maximize
        else:
            raise ValueError(f"Unknown method: {method}")
        
        scores[k] = score
        print(f"  K={k}: {score:.4f}")
    
    optimal_k = max(scores, key=scores.get)
    print(f"\nOptimal K: {optimal_k}")
    
    return optimal_k, scores


def visualize_clusters(
    embeddings: np.ndarray,
    cluster_assignments: np.ndarray,
    ground_truth_labels: Optional[np.ndarray] = None,
    domain_names: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    method: str = 'tsne'
):
    """
    Visualize discovered clusters.
    
    Args:
        embeddings: CLIP embeddings [N, D]
        cluster_assignments: Predicted cluster labels [N]
        ground_truth_labels: Optional ground truth domain labels [N]
        domain_names: Optional list of domain names
        output_path: Path to save visualization
        method: 'tsne' or 'umap'
    """
    print(f"Visualizing clusters using {method.upper()}...")
    
    # Dimensionality reduction
    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    elif method == 'umap':
        from umap import UMAP
        reducer = UMAP(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Create visualization
    if ground_truth_labels is not None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        axes = [axes]
    
    # Plot discovered clusters
    ax = axes[0]
    unique_clusters = np.unique(cluster_assignments)
    
    for cluster_id in unique_clusters:
        mask = cluster_assignments == cluster_id
        label = f"Cluster {cluster_id}" if cluster_id >= 0 else "Noise"
        alpha = 0.3 if cluster_id == -1 else 0.6
        
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            label=label,
            alpha=alpha,
            s=20
        )
    
    ax.set_xlabel(f'{method.upper()} 1', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{method.upper()} 2', fontsize=12, fontweight='bold')
    ax.set_title('Discovered Pseudo-Domains', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)
    
    # Plot ground truth if available
    if ground_truth_labels is not None:
        ax = axes[1]
        unique_gt = np.unique(ground_truth_labels)
        
        for gt_id in unique_gt:
            mask = ground_truth_labels == gt_id
            label = domain_names[gt_id] if domain_names else f"Domain {gt_id}"
            
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                label=label,
                alpha=0.6,
                s=20
            )
        
        ax.set_xlabel(f'{method.upper()} 1', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{method.upper()} 2', fontsize=12, fontweight='bold')
        ax.set_title('Ground Truth Domains', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Discover domains in CDDB dataset')
    parser.add_argument('--embeddings', type=str, required=True,
                       help='Path to cached CLIP embeddings (.npz file)')
    parser.add_argument('--method', type=str, default='kmeans',
                       choices=['kmeans', 'dbscan', 'agglomerative'],
                       help='Clustering method')
    parser.add_argument('--n_clusters', type=int, default=12,
                       help='Number of clusters (for kmeans/agglomerative)')
    parser.add_argument('--find_optimal_k', action='store_true',
                       help='Find optimal K using silhouette analysis')
    parser.add_argument('--visualize', action='store_true',
                       help='Create t-SNE visualization')
    parser.add_argument('--output_dir', type=str, default='domain_discovery',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load embeddings
    print(f"Loading embeddings from {args.embeddings}")
    data = np.load(args.embeddings)
    embeddings = data['embeddings']
    labels = data['labels']
    domain_indices = data['domain_indices']
    
    print(f"Loaded {len(embeddings)} embeddings")
    
    # Domain names
    domain_names = [
        'biggan', 'crn', 'cyclegan', 'deepfake', 'gaugan', 'glow',
        'imle', 'san', 'stargan_gf', 'stylegan', 'whichfaceisreal', 'wild'
    ]
    
    # Find optimal K if requested
    if args.find_optimal_k:
        optimal_k, scores = find_optimal_k(embeddings, k_range=range(2, 25))
        args.n_clusters = optimal_k
    
    # Discover domains
    discoverer = DomainDiscovery(
        method=args.method,
        n_clusters=args.n_clusters,
        random_state=42
    )
    
    cluster_assignments = discoverer.fit(embeddings)
    
    # Evaluate
    evaluator = ClusterEvaluator()
    metrics = evaluator.evaluate(
        embeddings=embeddings,
        cluster_assignments=cluster_assignments,
        ground_truth_labels=domain_indices
    )
    
    evaluator.print_metrics(metrics)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'method': args.method,
        'n_clusters': args.n_clusters,
        'metrics': metrics,
        'cluster_assignments': cluster_assignments.tolist()
    }
    
    with open(output_dir / 'discovery_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved results to {output_dir / 'discovery_results.json'}")
    
    # Visualize if requested
    if args.visualize:
        visualize_clusters(
            embeddings=embeddings,
            cluster_assignments=cluster_assignments,
            ground_truth_labels=domain_indices,
            domain_names=domain_names,
            output_path=str(output_dir / 'cluster_visualization.png'),
            method='tsne'
        )
    
    print("\n✓ Domain discovery complete!")
