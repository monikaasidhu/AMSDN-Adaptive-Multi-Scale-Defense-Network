"""
Self-Supervised Robustness Training (SSRT) Pretraining
Stage 5 of AMSDN pipeline
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.amsdn import AMSDNWithSSRT
from data.cifar10 import get_ssrt_loader


class SSRTTrainer:
    """Self-Supervised Robustness Training"""
    
    def __init__(self, model, device='cuda', lr=1e-4, save_dir='./checkpoints'):
        self.model = model.to(device)
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        
        # Loss functions
        self.reconstruction_loss = nn.MSELoss()
        self.contrastive_loss = nn.CosineSimilarity(dim=1)
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10, eta_min=1e-6
        )
        
        # Logging
        self.writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs'))
        
    def add_synthetic_perturbation(self, images, epsilon=0.05):
        """Add synthetic adversarial-like perturbation"""
        perturbation = torch.randn_like(images) * epsilon
        perturbed = torch.clamp(images + perturbation, -2, 2)  # Normalized range
        return perturbed
    
    def compute_loss(self, original, masked, reconstructed):
        """
        Compute SSRT loss:
        - Reconstruction loss on masked regions
        - Contrastive loss for robustness
        """
        # Reconstruction loss
        recon_loss = self.reconstruction_loss(reconstructed, original)
        
        # Contrastive loss: features should be similar for original and perturbed
        orig_features, _, _, _ = self.model.amsdn.forward_features(original)
        perturbed = self.add_synthetic_perturbation(original)
        pert_features, _, _, _ = self.model.amsdn.forward_features(perturbed)
        
        # Compute feature similarity at each scale
        contrastive_losses = []
        for orig_feat, pert_feat in zip(orig_features, pert_features):
            # Global average pooling
            orig_pooled = torch.nn.functional.adaptive_avg_pool2d(orig_feat, 1).flatten(1)
            pert_pooled = torch.nn.functional.adaptive_avg_pool2d(pert_feat, 1).flatten(1)
            
            # Cosine similarity (we want high similarity)
            similarity = self.contrastive_loss(orig_pooled, pert_pooled)
            contrastive_losses.append(1 - similarity.mean())  # Convert to loss
        
        contrastive_loss = torch.stack(contrastive_losses).mean()
        
        # Total loss
        total_loss = recon_loss + 0.1 * contrastive_loss
        
        return total_loss, recon_loss, contrastive_loss
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        total_recon = 0
        total_contrast = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (masked, original, _) in enumerate(pbar):
            masked = masked.to(self.device)
            original = original.to(self.device)
            
            # Forward pass
            reconstructed = self.model(masked, mode='reconstruct')
            
            # Compute loss
            loss, recon_loss, contrast_loss = self.compute_loss(
                original, masked, reconstructed
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_contrast += contrast_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'contrast': f'{contrast_loss.item():.4f}'
            })
            
            # Log to tensorboard
            step = epoch * len(train_loader) + batch_idx
            self.writer.add_scalar('Train/Loss', loss.item(), step)
            self.writer.add_scalar('Train/Reconstruction', recon_loss.item(), step)
            self.writer.add_scalar('Train/Contrastive', contrast_loss.item(), step)
        
        # Average metrics
        avg_loss = total_loss / len(train_loader)
        avg_recon = total_recon / len(train_loader)
        avg_contrast = total_contrast / len(train_loader)
        
        return avg_loss, avg_recon, avg_contrast
    
    def train(self, train_loader, num_epochs=10):
        """Full training loop"""
        print("=" * 60)
        print("Starting SSRT Pretraining")
        print("=" * 60)
        
        best_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            # Train
            avg_loss, avg_recon, avg_contrast = self.train_epoch(train_loader, epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            # Log epoch metrics
            print(f"\nEpoch {epoch}/{num_epochs}:")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  Reconstruction: {avg_recon:.4f}")
            print(f"  Contrastive: {avg_contrast:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            self.writer.add_scalar('Epoch/Loss', avg_loss, epoch)
            self.writer.add_scalar('Epoch/Reconstruction', avg_recon, epoch)
            self.writer.add_scalar('Epoch/Contrastive', avg_contrast, epoch)
            self.writer.add_scalar('Epoch/LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Save checkpoint
            if avg_loss < best_loss:
                best_loss = avg_loss
                checkpoint_path = os.path.join(self.save_dir, 'ssrt_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': avg_loss,
                }, checkpoint_path)
                print(f"  ✓ Saved best checkpoint (loss: {best_loss:.4f})")
            
            # Save periodic checkpoint
            if epoch % 10 == 0:
                checkpoint_path = os.path.join(self.save_dir, f'ssrt_epoch_{epoch}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': avg_loss,
                }, checkpoint_path)
                
                
        
        self.writer.close()
        print("\n" + "=" * 60)
        print("SSRT Pretraining Complete!")
        print("=" * 60)


def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    print("Loading CIFAR-10 dataset...")
    train_loader = get_ssrt_loader(
        data_dir='./data',
        batch_size=128,
        num_workers=2
    )
    
    # Model
    print("Initializing AMSDN with SSRT...")
    model = AMSDNWithSSRT(num_classes=10, pretrained=True)
    
    # Trainer
    trainer = SSRTTrainer(
        model=model,
        device=device,
        lr=1e-4,
        save_dir='./checkpoints/ssrt'
    )
    
    # Train
    trainer.train(train_loader, num_epochs=10)


if __name__ == "__main__":
    main()
