"""
Utility functions for AMSDN
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os


def denormalize_cifar(images, mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]):
    """
    Denormalize CIFAR-10 images for visualization
    Args:
        images: Normalized images [B, C, H, W]
    Returns:
        Denormalized images [B, C, H, W] in [0, 1]
    """
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(images.device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(images.device)
    
    images = images * std + mean
    images = torch.clamp(images, 0, 1)
    
    return images


def visualize_adversarial_examples(clean_images, adv_images, labels, preds_clean, preds_adv, 
                                   save_path=None, max_images=8):
    """
    Visualize clean vs adversarial examples
    Args:
        clean_images: Clean images [B, C, H, W]
        adv_images: Adversarial images [B, C, H, W]
        labels: True labels [B]
        preds_clean: Predictions on clean [B]
        preds_adv: Predictions on adversarial [B]
        save_path: Path to save figure
        max_images: Maximum number of images to show
    """
    B = min(clean_images.size(0), max_images)
    
    # Denormalize
    clean_denorm = denormalize_cifar(clean_images[:B])
    adv_denorm = denormalize_cifar(adv_images[:B])
    
    # Create figure
    fig, axes = plt.subplots(3, B, figsize=(B*2, 6))
    
    for i in range(B):
        # Clean image
        axes[0, i].imshow(clean_denorm[i].permute(1, 2, 0).cpu())
        axes[0, i].set_title(f'Clean\nTrue: {labels[i].item()}\nPred: {preds_clean[i].item()}')
        axes[0, i].axis('off')
        
        # Adversarial image
        axes[1, i].imshow(adv_denorm[i].permute(1, 2, 0).cpu())
        axes[1, i].set_title(f'Adversarial\nPred: {preds_adv[i].item()}')
        axes[1, i].axis('off')
        
        # Difference (amplified)
        diff = (adv_denorm[i] - clean_denorm[i]).abs()
        diff_amplified = diff * 10  # Amplify for visibility
        diff_amplified = torch.clamp(diff_amplified, 0, 1)
        axes[2, i].imshow(diff_amplified.permute(1, 2, 0).cpu())
        axes[2, i].set_title('Difference (x10)')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_attention_maps(images, attention_maps, save_path=None, max_images=4):
    """
    Visualize spatial attention maps
    Args:
        images: Input images [B, C, H, W]
        attention_maps: List of attention map dicts for each FPN level
        save_path: Path to save figure
        max_images: Maximum number of images to show
    """
    B = min(images.size(0), max_images)
    num_levels = len(attention_maps)
    
    # Denormalize images
    images_denorm = denormalize_cifar(images[:B])
    
    fig, axes = plt.subplots(B, num_levels + 1, figsize=((num_levels+1)*2, B*2))
    
    for i in range(B):
        # Original image
        axes[i, 0].imshow(images_denorm[i].permute(1, 2, 0).cpu())
        axes[i, 0].set_title('Original' if i == 0 else '')
        axes[i, 0].axis('off')
        
        # Attention maps at each level
        for level in range(num_levels):
            spatial_attn = attention_maps[level]['spatial'][i, 0].cpu().detach()
            
            # Resize to match image size
            import torch.nn.functional as F
            spatial_attn = F.interpolate(
                spatial_attn.unsqueeze(0).unsqueeze(0),
                size=images_denorm.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).squeeze()
            
            axes[i, level + 1].imshow(spatial_attn, cmap='hot')
            axes[i, level + 1].set_title(f'FPN Level {level+2}' if i == 0 else '')
            axes[i, level + 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def compute_perturbation_stats(clean_images, adv_images):
    """
    Compute statistics about adversarial perturbations
    Args:
        clean_images: Clean images [B, C, H, W]
        adv_images: Adversarial images [B, C, H, W]
    Returns:
        Dictionary of statistics
    """
    diff = adv_images - clean_images
    
    # L2 norm per image
    l2_norms = diff.view(diff.size(0), -1).norm(p=2, dim=1)
    
    # Linf norm per image
    linf_norms = diff.view(diff.size(0), -1).abs().max(dim=1)[0]
    
    # L0 norm (number of changed pixels) per image
    l0_norms = (diff.abs().sum(dim=1) > 0).float().sum(dim=(1, 2))
    
    stats = {
        'l2_mean': l2_norms.mean().item(),
        'l2_std': l2_norms.std().item(),
        'l2_max': l2_norms.max().item(),
        'linf_mean': linf_norms.mean().item(),
        'linf_max': linf_norms.max().item(),
        'l0_mean': l0_norms.mean().item(),
        'l0_std': l0_norms.std().item()
    }
    
    return stats


def save_checkpoint(model, optimizer, epoch, metrics, path):
    """
    Save training checkpoint
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Dictionary of metrics
        path: Save path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path, device='cuda'):
    """
    Load training checkpoint
    Args:
        model: Model to load into
        optimizer: Optimizer to load into
        path: Checkpoint path
        device: Device to map to
    Returns:
        epoch, metrics
    """
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})
    
    print(f"Checkpoint loaded from {path} (epoch {epoch})")
    
    return epoch, metrics


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self, name):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f'{self.name}: {self.avg:.4f}'


def set_random_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    # For reproducibility on CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'total_M': total / 1e6,
        'trainable_M': trainable / 1e6
    }