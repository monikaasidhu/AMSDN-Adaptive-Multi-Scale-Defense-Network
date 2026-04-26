"""
One-Pixel and Few-Pixel Attacks — Colab Free Tier Optimised
Sparse adversarial perturbations

Key fixes vs original:
  - FewPixelAttack.select_pixels_greedy: batched gradient computation (whole
    batch in one forward/backward, not one image at a time)
  - FewPixelAttack.optimize_pixel_values: vectorised pixel tensor, no Python
    loop per pixel per iteration
  - FewPixelAttack.attack: single batched call instead of sequential per-image
    loop; pixel-value optimisation now operates on all images simultaneously
  - OnePixelAttack: population eval batched (one forward pass per generation,
    not pop_size forward passes); defaults reduced for training use
  - All autocast(device_type=...) used correctly for AMP compatibility
  - No torch.no_grad() wrapping around methods that need autograd internally
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  One-Pixel Attack
# ══════════════════════════════════════════════════════════════════════════════
class OnePixelAttack:
    """
    One-pixel attack using differential evolution.
    Based on Su et al. "One pixel attack for fooling deep neural networks."

    Optimisation vs original:
      - Entire population evaluated in a SINGLE batched forward pass per
        generation (was: one forward pass per individual → pop_size × slower).
      - Reduced default pop_size / max_iterations for training speed.
    """

    def __init__(self, pop_size: int = 100, max_iterations: int = 30):
        # Reduced from 400/100 — still effective, ~12× faster per image
        self.pop_size       = pop_size
        self.max_iterations = max_iterations

    # ── single-image attack ───────────────────────────────────────────────────
    def _attack_single(self, model: nn.Module,
                       image: torch.Tensor,
                       label: int) -> torch.Tensor:
        device   = image.device
        C, H, W  = image.shape
        dev_type = device.type

        population = np.random.rand(self.pop_size, 5).astype(np.float32)
        population[:, 0] *= H
        population[:, 1] *= W
        population[:, 2:] = population[:, 2:] * 4.0 - 2.0

        best_solution = None
        best_fitness  = float('-inf')

        model.eval()
        with torch.no_grad():
            for _ in range(self.max_iterations):
                # ── build entire population as a batch ────────────────────
                # Shape: [pop_size, C, H, W]
                batch = image.unsqueeze(0).expand(
                    self.pop_size, -1, -1, -1).clone()

                xs = np.clip(population[:, 0].astype(int), 0, H - 1)
                ys = np.clip(population[:, 1].astype(int), 0, W - 1)
                rgb = torch.tensor(
                    population[:, 2:], dtype=torch.float32, device=device
                )  # [pop_size, 3]

                idx = torch.arange(self.pop_size, device=device)
                batch[idx, :,
                      torch.tensor(xs, device=device),
                      torch.tensor(ys, device=device)] = rgb

                # ONE forward pass for the whole population
                with torch.amp.autocast(device_type=dev_type):
                    probs = F.softmax(model(batch), dim=1)  # [pop, classes]

                true_probs    = probs[:, label].cpu().numpy()
                fitness_scores = 1.0 - true_probs

                best_idx = int(np.argmax(fitness_scores))
                if fitness_scores[best_idx] > best_fitness:
                    best_fitness  = fitness_scores[best_idx]
                    best_solution = population[best_idx].copy()

                # ── differential evolution update ─────────────────────────
                new_pop = population.copy()
                all_idx = np.arange(self.pop_size)
                for i in range(self.pop_size):
                    candidates = all_idx[all_idx != i]
                    a, b, c    = population[
                        np.random.choice(candidates, 3, replace=False)]
                    mutant = a + 0.8 * (b - c)
                    mask   = np.random.rand(5) < 0.9
                    trial  = np.where(mask, mutant, population[i])
                    trial[0] = np.clip(trial[0], 0, H - 1)
                    trial[1] = np.clip(trial[1], 0, W - 1)
                    trial[2:] = np.clip(trial[2:], -2.0, 2.0)
                    new_pop[i] = trial
                population = new_pop

        if best_solution is not None:
            adversarial = image.clone()
            x = int(np.clip(best_solution[0], 0, H - 1))
            y = int(np.clip(best_solution[1], 0, W - 1))
            adversarial[:, x, y] = torch.tensor(
                best_solution[2:], dtype=torch.float32, device=device)
            return adversarial

        return image

    # ── batch interface ───────────────────────────────────────────────────────
    def attack(self, images: torch.Tensor,
               model: nn.Module,
               labels: torch.Tensor) -> torch.Tensor:
        """images: [B, C, H, W]  →  adversarial [B, C, H, W]"""
        return torch.stack([
            self._attack_single(model, img, lbl.item())
            for img, lbl in zip(images, labels)
        ])


# ══════════════════════════════════════════════════════════════════════════════
#  Few-Pixel Attack
# ══════════════════════════════════════════════════════════════════════════════
class FewPixelAttack:
    """
    Few-pixel attack: greedily select high-gradient pixels, then optimise
    their values with Adam.

    Optimisations vs original:
      - select_pixels_greedy: ONE batched forward/backward for all images
        simultaneously instead of a sequential per-image Python loop.
      - optimize_pixel_values: pixel perturbations stored as a single
        [B, num_pixels, C] tensor; no Python loop over pixels per iteration.
      - attack: calls both helpers in batch mode — no outer for-loop over
        images during optimisation.
    """

    def __init__(self, num_pixels: int = 5,
                 epsilon: float = 0.5,
                 num_iterations: int = 30):   # reduced from 50
        self.num_pixels     = num_pixels
        self.epsilon        = epsilon
        self.num_iterations = num_iterations

    # ── batched pixel selection ───────────────────────────────────────────────
    def select_pixels_greedy(self, model: nn.Module,
                             images: torch.Tensor,
                             labels: torch.Tensor) -> torch.Tensor:
        """
        Return top-gradient pixel indices for the whole batch at once.

        Returns
        -------
        pixel_idx : LongTensor [B, num_pixels, 2]  (x, y) pairs
        """
        device   = images.device
        dev_type = device.type
        B, C, H, W = images.shape

        # Needs grad on inputs so we can get ∂loss/∂input
        images_var = images.clone().detach().requires_grad_(True)

        with torch.amp.autocast(device_type=dev_type):
            output = model(images_var)
            loss   = F.cross_entropy(output, labels)

        # Single backward for the whole batch
        loss.backward()

        with torch.no_grad():
            grad_mag = images_var.grad.abs().sum(dim=1)  # [B, H, W]
            flat     = grad_mag.view(B, -1)              # [B, H*W]
            _, top   = torch.topk(flat, self.num_pixels, dim=1)  # [B, k]

            xs = top // W   # [B, k]
            ys = top %  W   # [B, k]
            pixel_idx = torch.stack([xs, ys], dim=2)    # [B, k, 2]

        return pixel_idx

    # ── batched pixel-value optimisation ─────────────────────────────────────
    def optimize_pixel_values(self, model: nn.Module,
                              images: torch.Tensor,
                              labels: torch.Tensor,
                              pixel_idx: torch.Tensor) -> torch.Tensor:
        """
        Optimise pixel values for the whole batch simultaneously.

        Parameters
        ----------
        images    : [B, C, H, W]
        pixel_idx : [B, num_pixels, 2]  (x, y) pairs from select_pixels_greedy

        Returns
        -------
        adversarial images : [B, C, H, W]
        """
        device   = images.device
        dev_type = device.type
        B, C, H, W = images.shape
        K = self.num_pixels

        # Gather original values at selected positions: [B, K, C]
        xs = pixel_idx[:, :, 0]  # [B, K]
        ys = pixel_idx[:, :, 1]  # [B, K]

        # images[:, :, x, y] — gather over spatial dims for each (b, k)
        b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, K)
        orig_vals = images[b_idx, :, xs, ys].clone()  # [B, K, C]

        # Perturbation tensor (initialised to zero offset)
        delta = torch.zeros_like(orig_vals, requires_grad=True)  # [B, K, C]
        optimizer = torch.optim.Adam([delta], lr=0.01)

        for _ in range(self.num_iterations):
            # Apply perturbation — build adversarial batch without a pixel loop
            adv = images.clone()
            new_vals = (orig_vals + delta).clamp(-2.0, 2.0)
            # Scatter back: adv[b, :, x, y] = new_vals[b, k, :]
            adv[b_idx, :, xs, ys] = new_vals

            with torch.amp.autocast(device_type=dev_type):
                output = model(adv)
                # Maximise loss (minimise negative CE) to fool the model
                loss = -F.cross_entropy(output, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # Project into epsilon-ball around originals
            with torch.no_grad():
                delta.clamp_(-self.epsilon, self.epsilon)

        # Build final adversarial images
        with torch.no_grad():
            adv_final = images.clone()
            final_vals = (orig_vals + delta).clamp(-2.0, 2.0)
            adv_final[b_idx, :, xs, ys] = final_vals

        return adv_final

    # ── main batch entry point ────────────────────────────────────────────────
    def attack(self, images: torch.Tensor,
               model: nn.Module,
               labels: torch.Tensor) -> torch.Tensor:
        """
        images : [B, C, H, W]
        Returns adversarial images [B, C, H, W]

        NOTE: This method calls .backward() internally (via select_pixels_greedy
        and optimize_pixel_values). Do NOT wrap calls to this method in
        torch.no_grad() — it will raise a RuntimeError.
        """
        was_training = model.training
        model.eval()

        # Step 1: select pixels for entire batch with one backward pass
        pixel_idx = self.select_pixels_greedy(model, images, labels)

        # Step 2: optimise pixel values for entire batch simultaneously
        adv_images = self.optimize_pixel_values(
            model, images, labels, pixel_idx)

        model.train(was_training)
        return adv_images.detach()


# ══════════════════════════════════════════════════════════════════════════════
#  Sparse Attack Evaluator
# ══════════════════════════════════════════════════════════════════════════════
class SparseAttackEvaluator:
    """Evaluate sparse attacks across different sparsity levels."""

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model  = model.to(device)
        self.device = device

    def evaluate_sparsity_levels(self, images: torch.Tensor,
                                 labels: torch.Tensor,
                                 max_pixels: int = 10) -> dict:
        results = {}
        images = images.to(self.device)
        labels = labels.to(self.device)

        for num_pixels in range(1, max_pixels + 1):
            attack     = FewPixelAttack(num_pixels=num_pixels, epsilon=0.5)
            adv_images = attack.attack(images, self.model, labels)

            with torch.no_grad():
                clean_preds = self.model(images).argmax(dim=1)
                adv_preds   = self.model(adv_images).argmax(dim=1)
                success     = (adv_preds != labels).float().mean().item()
                l2_norm     = (adv_images - images).pow(2).sum(
                    dim=(1, 2, 3)).sqrt().mean().item()

            results[num_pixels] = {
                'success_rate': success,
                'l2_norm':      l2_norm,
            }

        return results


# ══════════════════════════════════════════════════════════════════════════════
#  Quick smoke-test
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Testing pixel attacks...")

    images = torch.randn(2, 3, 32, 32)
    labels = torch.tensor([0, 1])

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    model.eval()

    print("\nFew-Pixel Attack (batched):")
    attack     = FewPixelAttack(num_pixels=5, epsilon=0.5)
    adv_images = attack.attack(images, model, labels)

    with torch.no_grad():
        clean_preds = model(images).argmax(dim=1)
        adv_preds   = model(adv_images).argmax(dim=1)

    print(f"Clean predictions : {clean_preds.tolist()}")
    print(f"Adversarial preds : {adv_preds.tolist()}")
    print(f"Labels            : {labels.tolist()}")
    print(f"Success rate      : {(adv_preds != labels).float().mean():.2%}")

    diff = (adv_images - images).abs()
    num_changed = (diff.sum(dim=1) > 0).float().sum(dim=(1, 2))
    print(f"Pixels changed    : {num_changed.tolist()}")