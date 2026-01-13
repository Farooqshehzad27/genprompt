# GenPrompt: Unsupervised Domain Discovery for Continual Deepfake Detection

Implementation of GenPrompt framework for open-world continual deepfake detection on the CDDB dataset.

## Overview

GenPrompt automatically discovers deepfake generator domains through unsupervised clustering in CLIP embedding space and dynamically allocates domain-specific prompts for continual learning without catastrophic forgetting.

## Key Features

- **Unsupervised Domain Discovery (L3D):** Automatic clustering in CLIP space
- **Dynamic Prompt Allocation:** On-demand prompt creation for new domains
- **Prompt Routing Network (PRN):** Mixture-of-experts routing to domain-specific prompts
- **Cluster-Aligned Contrastive Learning:** Prototype-based learning for stability
- **Continual Learning:** Sequential domain arrival without replay

## Project Structure

```
genprompt/
├── data_loader.py              # CDDB dataset loader
├── clip_encoder.py             # CLIP feature extraction
├── domain_discovery.py         # Unsupervised clustering
├── prompt_pool.py              # Dynamic prompt management
├── prompt_router.py            # Prompt routing network
├── contrastive_loss.py         # Contrastive learning
├── train_genprompt.py          # Main training pipeline
├── dataset_analysis.py         # Dataset statistics
├── requirements.txt            # Dependencies
└── CDDB/                       # Dataset (not included)
```

## Installation

```bash
# Clone repository
git clone <your-repo-url>
cd genprompt

# Install dependencies
pip install -r requirements.txt

# Install CLIP
pip install git+https://github.com/openai/CLIP.git
```

## Dataset

Download the CDDB dataset and place it in the `CDDB/` directory with the following structure:

```
CDDB/
├── biggan/
│   ├── train/
│   │   ├── 0_real/
│   │   └── 1_fake/
│   └── val/
├── crn/
├── cyclegan/  # Has subdomains
├── ...
└── wild/
```

## Quick Start

### 1. Analyze Dataset

```bash
python dataset_analysis.py --data_root CDDB --output_dir dataset_analysis
```

### 2. Extract CLIP Embeddings (Optional - will be done automatically during training)

```bash
python clip_encoder.py --data_root CDDB --model ViT-B/32 --split train --visualize
```

### 3. Train GenPrompt

```bash
python train_genprompt.py \
  --data_root CDDB \
  --domains biggan crn cyclegan deepfake \
  --clip_model ViT-B/32 \
  --n_clusters 12 \
  --routing_mode soft \
  --batch_size 64 \
  --epochs_per_domain 10 \
  --use_cached
```

## Training Arguments

### Data
- `--data_root`: Path to CDDB dataset
- `--domains`: List of domains to train on (in order)

### Model
- `--clip_model`: CLIP model (`ViT-B/32` or `ViT-L/14`)
- `--n_clusters`: Number of clusters for domain discovery
- `--clustering_method`: Clustering algorithm (`kmeans`, `dbscan`, `agglomerative`)
- `--routing_mode`: Routing mode (`soft` or `hard`)

### Training
- `--batch_size`: Batch size (default: 64)
- `--epochs_per_domain`: Training epochs per domain (default: 10)
- `--lr_prompts`: Learning rate for prompts (default: 1e-3)
- `--lr_router`: Learning rate for router (default: 1e-4)
- `--lambda_cls`: Weight for classification loss (default: 1.0)
- `--lambda_con`: Weight for contrastive loss (default: 0.5)

### Output
- `--output_dir`: Output directory for experiments
- `--cache_dir`: Directory for cached CLIP embeddings
- `--use_cached`: Use cached embeddings if available

## Components

### Data Loader (`data_loader.py`)
- Handles standard and subdomain structures
- Supports PNG and JPEG formats
- Continual learning protocol with sequential domain arrival

### CLIP Encoder (`clip_encoder.py`)
- Extracts CLIP embeddings with caching
- Supports ViT-B/32 and ViT-L/14
- Includes visualization utilities

### Domain Discovery (`domain_discovery.py`)
- K-means, DBSCAN, and Agglomerative clustering
- Quality metrics: Silhouette, Davies-Bouldin, NMI, ARI
- Prototype computation and online updates

### Prompt Pool (`prompt_pool.py`)
- Visual prompts (VPT-style)
- Textual prompts (learnable context)
- Dynamic allocation for new clusters

### Prompt Router (`prompt_router.py`)
- Lightweight MLP router
- Soft (mixture-of-experts) and hard (argmax) routing
- Per-prompt expert classifiers

### Contrastive Learning (`contrastive_loss.py`)
- Cluster-aligned contrastive loss
- Prototype management with EMA updates
- Prevents catastrophic forgetting

## Experimental Results

Results will be added after Phase 4-5 experiments are complete.

## Citation

```bibtex
@article{genprompt2024,
  title={GenPrompt: Unsupervised Domain Discovery and Prompt Routing for Open-World Continual Deepfake Detection},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

MIT License

## Acknowledgments

- CLIP: OpenAI
- CDDB Dataset: [Citation]
- Inspired by L2P and DualPrompt continual learning methods
