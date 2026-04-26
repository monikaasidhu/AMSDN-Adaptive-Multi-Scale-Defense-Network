"""
Adversarial Patch Attacks
Physical-world style localized perturbations
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        patch = torch.rand(3, self.patch_size, self.patch_size, device=device)
        patch = (patch - 0.5) * 2 * self.epsilon
        return patch

    def apply_patch_to_image(self, image, patch, location=None):
        """
        Apply patch to image at specified or random location.
        Args:
            image: [C, H, W]
            patch: [C, patch_h, patch_w]
            location: (y, x) tuple or None for random
        Returns:
            patched image [C, H, W]
        """
        _, height, width = image.shape
        patch_h, patch_w = patch.shape[1:]

        if location is None:
            y = np.random.randint(0, height - patch_h + 1)
            x = np.random.randint(0, width - patch_w + 1)
        else:
            y, x = location

        patched = image.clone()
        patched[:, y:y + patch_h, x:x + patch_w] = patch
        return torch.clamp(patched, -2, 2)

    def optimize_patch(self, patch, images, model, labels):
        """
        Optimize patch to fool model.
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

        for _ in range(self.num_steps):
            patched_images = []
            for img in images:
                patched_images.append(self.apply_patch_to_image(img, patch))
            patched_batch = torch.stack(patched_images)

            outputs = model(patched_batch)
            loss = -F.cross_entropy(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                patch.clamp_(-self.epsilon, self.epsilon)

        return patch.detach()

    def apply(self, images, model, labels, optimize=True):
        """
        Apply patch attack to images.
        Args:
            images: [B, C, H, W]
            model: Target model
            labels: True labels
            optimize: If True, optimize patch; else use random
        Returns:
            adversarial images with patches
        """
        device = images.device

        if optimize:
            patch = self.initialize_patch(images.shape[-2:], device)
            patch = self.optimize_patch(patch, images, model, labels)
        else:
            patch = self.initialize_patch(images.shape[-2:], device)

        adversarial_images = []
        for img in images:
            adversarial_images.append(self.apply_patch_to_image(img, patch))

        return torch.stack(adversarial_images)


class PhysicalPatchTransform:
    """
    Simulate physical-world transformations for patch robustness
    """

    def __init__(self, rotation_range=30, scale_range=(0.8, 1.2)):
        self.rotation_range = rotation_range
        self.scale_range = scale_range

    def random_transform(self, patch):
        """Apply random transformation to patch."""
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        patch = self.rotate_patch(patch, angle)

        scale = np.random.uniform(*self.scale_range)
        patch = self.scale_patch(patch, scale)

        return patch

    def rotate_patch(self, patch, angle):
        """Rotate patch by angle (degrees)."""
        patch_batch = patch.unsqueeze(0)

        theta = torch.tensor([[
            [np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0],
            [np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0]
        ]], dtype=patch.dtype, device=patch.device)

        grid = F.affine_grid(theta, patch_batch.size(), align_corners=False)
        rotated = F.grid_sample(patch_batch, grid, align_corners=False)
        return rotated.squeeze(0)

    def scale_patch(self, patch, scale):
        """Scale patch."""
        _, height, width = patch.shape
        new_h, new_w = int(height * scale), int(width * scale)

        patch_batch = patch.unsqueeze(0)
        scaled = F.interpolate(
            patch_batch,
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        )

        if scale < 1:
            pad_h = (height - new_h) // 2
            pad_w = (width - new_w) // 2
            scaled = F.pad(scaled, (pad_w, pad_w, pad_h, pad_h))
        else:
            start_h = (new_h - height) // 2
            start_w = (new_w - width) // 2
            scaled = scaled[:, :, start_h:start_h + height, start_w:start_w + width]

        return scaled.squeeze(0)


class AdaptivePatchwithBPDA:
    """
    Adaptive patch attack using BPDA approximation.
    Attempts to bypass gradient obfuscation.
    """

    def __init__(self, patch_size=8, epsilon=0.5, num_steps=200):
        self.patch_attack = AdversarialPatch(patch_size, epsilon, num_steps)

    def bpda_forward(self, model, images):
        """
        Backward Pass through Differentiable Approximation.
        Replace non-differentiable defense components with identity.
        """
        fpn_features = model.backbone(images, return_features=True)
        attended_features, _ = model.attention(fpn_features)

        pooled_features = []
        for feat in attended_features:
            pooled = model.global_pool(feat).flatten(1)
            pooled_features.append(pooled)

        combined = torch.cat(pooled_features, dim=1)
        return model.classifier(combined)

    def attack(self, images, model, labels):
        """Adaptive attack using BPDA."""
        device = images.device
        patch = self.patch_attack.initialize_patch(images.shape[-2:], device)
        patch.requires_grad = True

        optimizer = torch.optim.Adam([patch], lr=0.01)

        for _ in range(self.patch_attack.num_steps):
            patched_images = []
            for img in images:
                patched_images.append(
                    self.patch_attack.apply_patch_to_image(img, patch)
                )
            patched_batch = torch.stack(patched_images)

            outputs = self.bpda_forward(model, patched_batch)
            loss = -F.cross_entropy(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                patch.clamp_(
                    -self.patch_attack.epsilon, self.patch_attack.epsilon
                )

        adversarial_images = []
        for img in images:
            adversarial_images.append(
                self.patch_attack.apply_patch_to_image(img, patch.detach())
            )

        return torch.stack(adversarial_images)


__all__ = ["AdversarialPatch", "AdaptivePatchwithBPDA", "PhysicalPatchTransform"]
