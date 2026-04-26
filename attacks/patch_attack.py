"""
Adversarial Patch Attacks
Physical-world style localized perturbations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AdversarialPatch:
    """
    Adversarial patch attack (simplified)
    Based on Brown et al. "Adversarial Patch"
    """
    
    def __init__(self, patch_size=8, epsilon=0.5, num_steps=100, lr=0.01):
        self.patch_size = patch_size
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.lr = lr
    
    def initialize_patch(self, image_size, device):
        """Initialize random patch"""
        # Random noise patch
        patch = torch.rand(3, self.patch_size, self.patch_size, device=device)
        patch = (patch - 0.5) * 2 * self.epsilon  # Scale to [-epsilon, epsilon]
        return patch
    
    def apply_patch_to_image(self, image, patch, location=None):
        """
        Apply patch to image at specified or random location
        Args:
            image: [C, H, W]
            patch: [C, patch_h, patch_w]
            location: (y, x) tuple or None for random
        Returns:
            patched image [C, H, W]
        """
        _, H, W = image.shape
        patch_h, patch_w = patch.shape[1:]
        
        if location is None:
            # Random location
            y = np.random.randint(0, H - patch_h + 1)
            x = np.random.randint(0, W - patch_w + 1)
        else:
            y, x = location
        
        # Clone image and apply patch
        patched = image.clone()
        patched[:, y:y+patch_h, x:x+patch_w] = patch
        
        # Clamp to valid range
        patched = torch.clamp(patched, -2, 2)
        
        return patched
    
    def optimize_patch(self, patch, images, model, labels):
        """
        Optimize patch to fool model
        Args:
            patch: [C, patch_h, patch_w]
            images: [B, C, H, W]
            model: Target model
            labels: True labels [B]
        Returns:
            optimized patch
        """
        patch = patch.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([patch], lr=self.lr)
        
        for step in range(self.num_steps):
            # Apply patch to all images at random locations
            patched_images = []
            for img in images:
                patched = self.apply_patch_to_image(img, patch)
                patched_images.append(patched)
            patched_batch = torch.stack(patched_images)
            
            # Forward pass
            outputs = model(patched_batch)
            
            # Loss: maximize loss for true labels (untargeted)
            loss = -F.cross_entropy(outputs, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Clip patch to valid perturbation range
            with torch.no_grad():
                patch.clamp_(-self.epsilon, self.epsilon)
        
        return patch.detach()
    
    def apply(self, images, model, labels, optimize=True):
        """
        Apply patch attack to images
        Args:
            images: [B, C, H, W]
            model: Target model
            labels: True labels
            optimize: If True, optimize patch; else use random
        Returns:
            adversarial images with patches
        """
        device = images.device
        B = images.size(0)
        
        if optimize:
            # Optimize universal patch for this batch
            patch = self.initialize_patch(images.shape[-2:], device)
            patch = self.optimize_patch(patch, images, model, labels)
        else:
            # Just use random patch
            patch = self.initialize_patch(images.shape[-2:], device)
        
        # Apply patch to all images
        adversarial_images = []
        for img in images:
            patched = self.apply_patch_to_image(img, patch)
            adversarial_images.append(patched)
        
        return torch.stack(adversarial_images)


class PhysicalPatchTransform:
    """
    Simulate physical-world transformations for patch robustness
    """
    
    def __init__(self, rotation_range=30, scale_range=(0.8, 1.2)):
        self.rotation_range = rotation_range
        self.scale_range = scale_range
    
    def random_transform(self, patch):
        """
        Apply random transformation to patch
        Args:
            patch: [C, H, W]
        Returns:
            transformed patch [C, H, W]
        """
        # Random rotation
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        patch = self.rotate_patch(patch, angle)
        
        # Random scale
        scale = np.random.uniform(*self.scale_range)
        patch = self.scale_patch(patch, scale)
        
        return patch
    
    def rotate_patch(self, patch, angle):
        """Rotate patch by angle (degrees)"""
        # Convert to batch format for grid_sample
        patch_batch = patch.unsqueeze(0)  # [1, C, H, W]
        
        # Create rotation matrix
        theta = torch.tensor([[
            [np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0],
            [np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0]
        ]], dtype=patch.dtype, device=patch.device)
        
        # Generate grid and sample
        grid = F.affine_grid(theta, patch_batch.size(), align_corners=False)
        rotated = F.grid_sample(patch_batch, grid, align_corners=False)
        
        return rotated.squeeze(0)
    
    def scale_patch(self, patch, scale):
        """Scale patch"""
        C, H, W = patch.shape
        new_H, new_W = int(H * scale), int(W * scale)
        
        # Resize
        patch_batch = patch.unsqueeze(0)
        scaled = F.interpolate(
            patch_batch, 
            size=(new_H, new_W), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Pad or crop to original size
        if scale < 1:
            # Pad
            pad_h = (H - new_H) // 2
            pad_w = (W - new_W) // 2
            scaled = F.pad(scaled, (pad_w, pad_w, pad_h, pad_h))
        else:
            # Crop
            start_h = (new_H - H) // 2
            start_w = (new_W - W) // 2
            scaled = scaled[:, :, start_h:start_h+H, start_w:start_w+W]
        
        return scaled.squeeze(0)


# Adaptive attack that tries to bypass AMSDN defenses
class AdaptivePatchwithBPDA:
    """
    Adaptive patch attack using BPDA approximation
    Attempts to bypass gradient obfuscation
    """
    
    def __init__(self, patch_size=8, epsilon=0.5, num_steps=200):
        self.patch_attack = AdversarialPatch(patch_size, epsilon, num_steps)
    
    def bpda_forward(self, model, images):
        """
        Backward Pass through Differentiable Approximation
        Replace non-differentiable defense components with identity
        """
        # Get features before purification
        fpn_features = model.backbone(images, return_features=True)
        attended_features, _ = model.attention(fpn_features)
        
        # Skip purification (treat as identity for gradients)
        # This approximates the gradient
        
        # Continue with classification
        pooled_features = []
        for feat in attended_features:
            pooled = model.global_pool(feat).flatten(1)
            pooled_features.append(pooled)
        
        combined = torch.cat(pooled_features, dim=1)
        logits = model.classifier(combined)
        
        return logits
    
    def attack(self, images, model, labels):
        """
        Adaptive attack using BPDA
        """
        device = images.device
        patch = self.patch_attack.initialize_patch(images.shape[-2:], device)
        patch.requires_grad = True
        
        optimizer = torch.optim.Adam([patch], lr=0.01)
        
        for step in range(self.patch_attack.num_steps):
            # Apply patch
            patched_images = []
            for img in images:
                patched = self.patch_attack.apply_patch_to_image(img, patch)
                patched_images.append(patched)
            patched_batch = torch.stack(patched_images)
            
            # BPDA forward (approximate gradients)
            outputs = self.bpda_forward(model, patched_batch)
            
            # Loss
            loss = -F.cross_entropy(outputs, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Clip
            with torch.no_grad():
                patch.clamp_(-self.patch_attack.epsilon, self.patch_attack.epsilon)
        
        # Apply final patch
        adversarial_images = []
        for img in images:
            patched = self.patch_attack.apply_patch_to_image(img, patch.detach())
            adversarial_images.append(patched)
        
        return torch.stack(adversarial_images)


# Test
if __name__ == "__main__":
    # Test basic patch attack
    patch_attack = AdversarialPatch(patch_size=4, epsilon=0.3)
    
    # Dummy data
    images = torch.randn(4, 3, 32, 32)
    labels = torch.randint(0, 10, (4,))
    
    # Dummy model
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 10)
    )
    
    # Apply attack
    adv_images = patch_attack.apply(images, model, labels, optimize=False)
    print(f"Original: {images.shape}, Adversarial: {adv_images.shape}")
    print(f"Perturbation: {(adv_images - images).abs().max().item():.4f}")