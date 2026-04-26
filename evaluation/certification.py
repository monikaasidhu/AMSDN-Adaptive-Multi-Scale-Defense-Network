"""
Randomized Smoothing Certification for AMSDN
Stage 6 of AMSDN pipeline (Corrected & Optimized)
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from scipy.stats import norm, beta
import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.amsdn import AMSDN
from data.cifar10 import CIFAR10DataModule


# ============================================================
# RANDOMIZED SMOOTHING CORE
# ============================================================

class RandomizedSmoothing:

    def __init__(self, model, num_classes=10, sigma=0.25, device='cuda'):
        self.model = model.to(device)
        self.num_classes = num_classes
        self.sigma = sigma
        self.device = device

    # --------------------------------------------------------

    def _predict_with_noise(self, x, num_samples):
        """
        Monte Carlo sampling with Gaussian noise
        Returns vote counts
        """
        B = x.size(0)
        class_counts = torch.zeros(B, self.num_classes, device=self.device)

        was_training = self.model.training
        self.model.eval()

        with torch.no_grad():
            for _ in range(num_samples):
                noise = torch.randn_like(x) * self.sigma
                noised = torch.clamp(x + noise, -2, 2)  # Clamp to valid range

                outputs = self.model(noised)
                preds = outputs.argmax(dim=1)

                # Vectorized vote accumulation
                class_counts.scatter_add_(
                    1,
                    preds.unsqueeze(1),
                    torch.ones_like(preds, dtype=torch.float).unsqueeze(1)
                )

        self.model.train(was_training)
        return class_counts

    # --------------------------------------------------------

    def certify(self, x, n0=100, n=10000, alpha=0.001):
        """
        Certify a single image.
        n0 = samples for class selection
        n  = samples for certification
        """

        # Step 1: Select most likely class
        counts_selection = self._predict_with_noise(x, n0)
        top_class = counts_selection.argmax(dim=1).item()

        # Step 2: Estimate probability with more samples
        counts_cert = self._predict_with_noise(x, n)
        nA = counts_cert[0, top_class].item()

        # Lower confidence bound (Clopper-Pearson)
        pA_lower = beta.ppf(alpha, nA, n - nA + 1)

        if pA_lower <= 0.5:
            return -1, 0.0  # Abstain

        # Certified L2 radius
        radius = self.sigma * norm.ppf(max(pA_lower, 1e-10))
        return top_class, float(radius)

    # --------------------------------------------------------

    def predict_batch(self, x, num_samples=100):
        """
        Smoothed prediction without certification
        """
        counts = self._predict_with_noise(x, num_samples)
        preds = counts.argmax(dim=1)
        probs = counts / num_samples
        return preds, probs


# ============================================================
# CERTIFICATION EVALUATOR
# ============================================================

class CertificationEvaluator:

    def __init__(self, model, sigma=0.25, device='cuda'):
        self.smoothed = RandomizedSmoothing(
            model=model,
            num_classes=10,
            sigma=sigma,
            device=device
        )
        self.device = device

    # --------------------------------------------------------

    def evaluate_certified_accuracy(self, test_loader,
                                    max_samples=100,
                                    n0=50,
                                    n=1000,
                                    alpha=0.001):

        certified_classes = []
        certified_radii = []
        true_labels = []

        sample_count = 0

        for images, labels in tqdm(test_loader, desc="Certifying"):

            images = images.to(self.device)
            labels = labels.to(self.device)

            for img, label in zip(images, labels):

                if sample_count >= max_samples:
                    break

                img = img.unsqueeze(0)

                pred, radius = self.smoothed.certify(
                    img,
                    n0=n0,
                    n=n,
                    alpha=alpha
                )

                certified_classes.append(pred)
                certified_radii.append(radius)
                true_labels.append(label.item())

                sample_count += 1

            if sample_count >= max_samples:
                break

        certified_classes = np.array(certified_classes)
        certified_radii = np.array(certified_radii)
        true_labels = np.array(true_labels)

        correct = (certified_classes == true_labels)

        clean_acc = correct.mean() * 100
        abstain_rate = (certified_classes == -1).mean() * 100

        radii_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
        certified_accs = {}

        for r in radii_levels:
            certified_accs[f"radius_{r}"] = (
                ((certified_radii >= r) & correct).mean() * 100
            )

        positive_radii = certified_radii[certified_radii > 0]

        results = {
            "clean_accuracy": clean_acc,
            "abstain_rate": abstain_rate,
            "certified_accuracies": certified_accs,
            "avg_certified_radius":
                float(positive_radii.mean()) if len(positive_radii) > 0 else 0.0,
            "median_certified_radius":
                float(np.median(positive_radii)) if len(positive_radii) > 0 else 0.0
        }

        return results

    # --------------------------------------------------------

    def print_results(self, results):

        print("\n" + "=" * 60)
        print("CERTIFICATION RESULTS")
        print("=" * 60)

        print(f"Clean Accuracy: {results['clean_accuracy']:.2f}%")
        print(f"Abstention Rate: {results['abstain_rate']:.2f}%\n")

        for k, v in results["certified_accuracies"].items():
            r = float(k.split("_")[1])
            print(f"L2 Radius {r:.2f}: {v:.2f}%")

        print(f"\nAverage Certified Radius: {results['avg_certified_radius']:.4f}")
        print(f"Median Certified Radius: {results['median_certified_radius']:.4f}")
        print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_module = CIFAR10DataModule(batch_size=32, num_workers=2)
    _, test_loader = data_module.get_loaders()

    model = AMSDN(num_classes=10, pretrained=False)

    checkpoint_candidates = [
        "./checkpoints/fast_robust_low_resource/amsdn_fast_robust_best.pth",
        "./checkpoints/fast_robust/amsdn_fast_robust_best.pth",
        "./checkpoints/finetuned/amsdn_finetuned_best.pth",
        "./checkpoints/adversarial/amsdn_best.pth",
        "./checkpoints/adversarial/amsdn_last.pth",
        "./checkpoints/amsdn_best.pth",
        "./checkpoints/amsdn_last.pth",
    ]
    checkpoint_path = None
    for path in checkpoint_candidates:
        if os.path.exists(path):
            checkpoint_path = path
            break

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("Warning: No trained model found.")

    evaluator = CertificationEvaluator(model, sigma=0.25, device=device)

    results = evaluator.evaluate_certified_accuracy(
        test_loader,
        max_samples=50,
        n0=50,
        n=1000,
        alpha=0.001
    )

    evaluator.print_results(results)

    os.makedirs("./results", exist_ok=True)
    with open("./results/certification_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n✓ Results saved to ./results/certification_results.json")


if __name__ == "__main__":
    main()