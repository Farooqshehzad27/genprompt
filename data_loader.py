"""
Data Loader for CDDB Dataset
Handles standard and subdomain structures, supports PNG and JPEG formats.
"""

import os
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CDDBDataset(Dataset):
    """
    CDDB Dataset loader with support for:
    - Standard and subdomain structures
    - PNG and JPEG formats
    - Continual learning protocols
    """
    
    def __init__(
        self,
        data_root: str,
        domains: List[str],
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        return_domain_label: bool = False
    ):
        """
        Args:
            data_root: Path to CDDB root directory
            domains: List of domain names to load
            split: 'train' or 'val'
            transform: Image transformations
            return_domain_label: If True, return (image, label, domain_id)
        """
        self.data_root = Path(data_root)
        self.domains = domains
        self.split = split
        self.transform = transform
        self.return_domain_label = return_domain_label
        
        # Domains with subdomain structures
        self.subdomain_domains = ['cyclegan', 'glow', 'stargan_gf', 'stylegan']
        
        # Build dataset index
        self.samples = []
        self.domain_to_idx = {domain: idx for idx, domain in enumerate(domains)}
        
        self._build_index()
        
    def _build_index(self):
        """Build index of all samples."""
        for domain in self.domains:
            domain_path = self.data_root / domain / self.split
            
            if not domain_path.exists():
                print(f"Warning: {domain_path} does not exist, skipping")
                continue
            
            domain_idx = self.domain_to_idx[domain]
            
            # Check if domain has subdomain structure
            if domain in self.subdomain_domains:
                self._index_subdomain_structure(domain_path, domain, domain_idx)
            else:
                self._index_standard_structure(domain_path, domain, domain_idx)
    
    def _index_standard_structure(self, domain_path: Path, domain: str, domain_idx: int):
        """Index standard domain structure (0_real, 1_fake)."""
        for class_name in ['0_real', '1_fake']:
            class_path = domain_path / class_name
            
            if not class_path.exists():
                continue
            
            label = 0 if class_name == '0_real' else 1
            
            # Get all image files (PNG and JPEG)
            for img_file in class_path.iterdir():
                if img_file.is_file() and img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    self.samples.append({
                        'path': str(img_file),
                        'label': label,
                        'domain': domain,
                        'domain_idx': domain_idx,
                        'subdomain': None
                    })
    
    def _index_subdomain_structure(self, domain_path: Path, domain: str, domain_idx: int):
        """Index domain with subdomain structure."""
        # Get all subdomain directories
        subdomains = [d for d in domain_path.iterdir() 
                     if d.is_dir() and d.name not in ['0_real', '1_fake']]
        
        for subdomain_path in subdomains:
            subdomain_name = subdomain_path.name
            
            for class_name in ['0_real', '1_fake']:
                class_path = subdomain_path / class_name
                
                if not class_path.exists():
                    continue
                
                label = 0 if class_name == '0_real' else 1
                
                # Get all image files
                for img_file in class_path.iterdir():
                    if img_file.is_file() and img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                        self.samples.append({
                            'path': str(img_file),
                            'label': label,
                            'domain': domain,
                            'domain_idx': domain_idx,
                            'subdomain': subdomain_name
                        })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        img = Image.open(sample['path']).convert('RGB')
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        label = sample['label']
        
        if self.return_domain_label:
            return img, label, sample['domain_idx']
        else:
            return img, label
    
    def get_domain_samples(self, domain: str) -> List[int]:
        """Get indices of all samples from a specific domain."""
        return [i for i, s in enumerate(self.samples) if s['domain'] == domain]
    
    def get_subdomain_samples(self, domain: str, subdomain: str) -> List[int]:
        """Get indices of all samples from a specific subdomain."""
        return [i for i, s in enumerate(self.samples) 
                if s['domain'] == domain and s['subdomain'] == subdomain]
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        stats = {
            'total_samples': len(self.samples),
            'domains': {},
            'class_distribution': {0: 0, 1: 0}
        }
        
        for sample in self.samples:
            domain = sample['domain']
            label = sample['label']
            
            if domain not in stats['domains']:
                stats['domains'][domain] = {
                    'total': 0,
                    'real': 0,
                    'fake': 0,
                    'subdomains': set()
                }
            
            stats['domains'][domain]['total'] += 1
            stats['domains'][domain][('real' if label == 0 else 'fake')] += 1
            
            if sample['subdomain']:
                stats['domains'][domain]['subdomains'].add(sample['subdomain'])
            
            stats['class_distribution'][label] += 1
        
        # Convert sets to lists for JSON serialization
        for domain in stats['domains']:
            stats['domains'][domain]['subdomains'] = sorted(list(stats['domains'][domain]['subdomains']))
        
        return stats


class ContinualCDDBDataset:
    """
    Wrapper for continual learning on CDDB dataset.
    Provides sequential domain arrival.
    """
    
    def __init__(
        self,
        data_root: str,
        domain_order: List[str],
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        return_domain_label: bool = False
    ):
        """
        Args:
            data_root: Path to CDDB root directory
            domain_order: Ordered list of domains for continual learning
            split: 'train' or 'val'
            transform: Image transformations
            return_domain_label: If True, return domain labels
        """
        self.data_root = data_root
        self.domain_order = domain_order
        self.split = split
        self.transform = transform
        self.return_domain_label = return_domain_label
        
        self.current_domain_idx = 0
        self.seen_domains = []
    
    def get_next_domain_dataset(self) -> Optional[CDDBDataset]:
        """Get dataset for the next domain in sequence."""
        if self.current_domain_idx >= len(self.domain_order):
            return None
        
        domain = self.domain_order[self.current_domain_idx]
        self.seen_domains.append(domain)
        self.current_domain_idx += 1
        
        dataset = CDDBDataset(
            data_root=self.data_root,
            domains=[domain],
            split=self.split,
            transform=self.transform,
            return_domain_label=self.return_domain_label
        )
        
        return dataset
    
    def get_all_seen_domains_dataset(self) -> CDDBDataset:
        """Get dataset containing all seen domains so far."""
        return CDDBDataset(
            data_root=self.data_root,
            domains=self.seen_domains,
            split=self.split,
            transform=self.transform,
            return_domain_label=self.return_domain_label
        )
    
    def reset(self):
        """Reset to beginning of domain sequence."""
        self.current_domain_idx = 0
        self.seen_domains = []


def get_clip_transforms(image_size: int = 224):
    """Get CLIP-compatible image transforms."""
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])


def get_default_transforms(image_size: int = 224, augment: bool = False):
    """Get default image transforms with optional augmentation."""
    if augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


if __name__ == "__main__":
    # Test the data loader
    data_root = r"d:\Farooq\genprompt\CDDB"
    
    # Test standard dataset
    print("Testing CDDBDataset...")
    dataset = CDDBDataset(
        data_root=data_root,
        domains=['biggan', 'cyclegan', 'whichfaceisreal'],
        split='train',
        transform=get_clip_transforms(),
        return_domain_label=True
    )
    
    print(f"Total samples: {len(dataset)}")
    
    # Get statistics
    stats = dataset.get_statistics()
    print("\nDataset Statistics:")
    print(json.dumps(stats, indent=2))
    
    # Test sample loading
    print("\nTesting sample loading...")
    img, label, domain_idx = dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Label: {label}")
    print(f"Domain idx: {domain_idx}")
    
    # Test continual learning wrapper
    print("\n" + "="*80)
    print("Testing ContinualCDDBDataset...")
    
    continual_dataset = ContinualCDDBDataset(
        data_root=data_root,
        domain_order=['biggan', 'crn', 'cyclegan'],
        split='train',
        transform=get_clip_transforms(),
        return_domain_label=True
    )
    
    # Simulate continual learning
    for i in range(3):
        domain_dataset = continual_dataset.get_next_domain_dataset()
        if domain_dataset:
            print(f"\nDomain {i+1}: {continual_dataset.seen_domains[-1]}")
            print(f"  Samples: {len(domain_dataset)}")
            
            # Test DataLoader
            loader = DataLoader(domain_dataset, batch_size=32, shuffle=True, num_workers=0)
            batch = next(iter(loader))
            print(f"  Batch shapes: {batch[0].shape}, {batch[1].shape}, {batch[2].shape}")
    
    print("\n✓ Data loader tests passed!")
