"""
Comprehensive Evaluation of AMSDN
Clean accuracy + Robust accuracy + Detection metrics + Adaptive attack

Fixes vs original:
  FIX 1 — Checkpoint path updated to also check fast_fgsm directory
  FIX 2 — Pixel attacks call .backward() internally; removed torch.no_grad()
           wrapping around their generation (same fix as finetune_attacks.py)
  FIX 3 — autocast(device_type=...) used everywhere instead of bare torch.no_grad
           to match AMP training — avoids FP16 inference inconsistency
  FIX 4 — device passed as string consistently ('cuda'/'cpu') not torch.device object
           because autocast requires device_type string
  FIX 5 — FewPixelAttack num_iterations reduced (30) for eval speed on Colab T4
"""

import torch
import torch.nn.functional as F
from torch.amp import autocast
import time
import os
import sys
import json
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.amsdn import AMSDN
from data.cifar10 import CIFAR10DataModule
from attacks.patch_attacks import AdversarialPatch, AdaptivePatchwithBPDA
from attacks.pixel_attacks import FewPixelAttack
from training.adversarial_train import PGDAttack


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluator
# ══════════════════════════════════════════════════════════════════════════════
class AMSDNEvaluator:

    def __init__(self, model, device: str = 'cuda', target_fpr: float = 5.0):
        # FIX 4: store device as plain string for autocast compatibility
        self.device = device if isinstance(device, str) else str(device)
        self.model  = model.to(self.device)
        self.model.eval()
        self.target_fpr = target_fpr
        self.detection_threshold = None

        # Attack suite — FIX 5: pixel attacks use num_iterations=30 for speed
        self.attacks = {
            'PGD-8':    PGDAttack(epsilon=8/255,  alpha=2/255, num_steps=20),
            'PGD-16':   PGDAttack(epsilon=16/255, alpha=2/255, num_steps=20),
            'Patch-4':  AdversarialPatch(patch_size=4, epsilon=0.3),
            'Patch-8':  AdversarialPatch(patch_size=8, epsilon=0.5),
            'Pixel-5':  FewPixelAttack(num_pixels=5,  epsilon=0.5, num_iterations=30),
            'Pixel-10': FewPixelAttack(num_pixels=10, epsilon=0.5, num_iterations=30),
        }

    # ── helpers ───────────────────────────────────────────────────────────────
    def _needs_grad(self, attack_name: str) -> bool:
        """
        Returns True for attacks that call .backward() internally.
        These must NOT be wrapped in torch.no_grad().
        """
        return attack_name.startswith('Pixel') or attack_name.startswith('Patch')

    # ── clean evaluation ──────────────────────────────────────────────────────
    def _collect_clean_anomaly_scores(self, test_loader, max_batches=None):
        """Collect anomaly scores on clean data for threshold calibration."""
        self.model.eval()
        scores = []
        batch_count = 0

        with torch.no_grad():
            for images, _ in tqdm(test_loader, desc='Calibrating threshold'):
                images = images.to(self.device, non_blocking=True)
                with autocast(device_type=self.device):
                    outputs = self.model(images, return_detailed=True)
                scores.append(outputs['avg_anomaly_score'].float().view(-1).cpu())

                batch_count += 1
                if max_batches and batch_count >= max_batches:
                    break

        if not scores:
            raise RuntimeError('No clean anomaly scores collected for calibration.')

        return torch.cat(scores, dim=0)

    def calibrate_detection_threshold(self, test_loader, max_batches=20):
        """
        Set the anomaly threshold to approximately match the requested clean FPR.
        """
        clean_scores = self._collect_clean_anomaly_scores(
            test_loader, max_batches=max_batches
        )
        quantile = min(max(1.0 - self.target_fpr / 100.0, 0.0), 1.0)
        self.detection_threshold = torch.quantile(clean_scores, quantile).item()
        return self.detection_threshold

    def _detect_from_scores(self, anomaly_scores):
        if self.detection_threshold is None:
            raise RuntimeError('Detection threshold not calibrated before evaluation.')
        return anomaly_scores.float().view(-1) > self.detection_threshold

    def evaluate_clean(self, test_loader) -> dict:
        self.model.eval()

        correct       = 0
        total         = 0
        total_time    = 0.0
        false_positive = 0

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Clean Evaluation'):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                start = time.time()
                # FIX 3: use autocast to match training precision
                with autocast(device_type=self.device):
                    outputs = self.model(images, return_detailed=True)
                total_time += time.time() - start

                preds = outputs['logits'].float().argmax(dim=1)
                detected = self._detect_from_scores(outputs['avg_anomaly_score'])
                correct        += (preds == labels).sum().item()
                false_positive += detected.sum().item()
                total          += labels.size(0)

        accuracy   = 100.0 * correct / total
        avg_time   = total_time / total
        throughput = total / total_time
        fpr        = 100.0 * false_positive / total

        return {
            'accuracy':               accuracy,
            'avg_inference_time_sec': avg_time,
            'throughput_img_per_sec': throughput,
            'false_positive_rate':    fpr,
            'detection_threshold':    self.detection_threshold,
        }

    # ── per-attack evaluation ─────────────────────────────────────────────────
    def evaluate_attack(self, test_loader, attack_name: str,
                        attack, max_batches=None) -> dict:
        self.model.eval()

        correct     = 0
        detected_count_total = 0
        total       = 0
        batch_count = 0

        for images, labels in tqdm(test_loader,
                                   desc=f'{attack_name} Evaluation'):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # ── generate adversarial examples ─────────────────────────────
            # FIX 2: Pixel and Patch attacks call .backward() internally —
            # do NOT wrap in torch.no_grad(). PGD is safe under no_grad
            # because we call attack.generate() which handles its own grads.
            if attack_name.startswith('PGD'):
                adv_images = attack.generate(self.model, images, labels)

            elif attack_name.startswith('Patch'):
                # AdversarialPatch.apply needs autograd — no no_grad here
                adv_images = attack.apply(images, self.model, labels)

            else:
                # FewPixelAttack.attack calls select_pixels_greedy which
                # calls loss.backward() — must NOT be under no_grad
                adv_images = attack.attack(images, self.model, labels)

            # ── evaluate model on adversarial examples ─────────────────────
            with torch.no_grad():
                with autocast(device_type=self.device):
                    outputs = self.model(adv_images.detach(),
                                        return_detailed=True)
                detected_mask = self._detect_from_scores(outputs['avg_anomaly_score'])
                preds     = outputs['logits'].float().argmax(dim=1)
                correct  += (preds == labels).sum().item()
                detected_count_total += detected_mask.sum().item()
                total    += labels.size(0)

            batch_count += 1
            if max_batches and batch_count >= max_batches:
                break

        robust_accuracy    = 100.0 * correct / total
        attack_success_rate = 100.0 * (1 - correct / total)
        detection_tpr       = 100.0 * detected_count_total / total

        return {
            'robust_accuracy':              robust_accuracy,
            'attack_success_rate':          attack_success_rate,
            'detection_true_positive_rate': detection_tpr,
        }

    # ── adaptive attack evaluation ────────────────────────────────────────────
    def evaluate_adaptive_attack(self, test_loader, max_batches=10) -> dict:
        self.model.eval()
        # AdaptivePatchwithBPDA also calls backward internally
        adaptive_attack = AdaptivePatchwithBPDA(patch_size=8, epsilon=0.5)

        correct     = 0
        detected_count_total = 0
        total       = 0
        batch_count = 0

        for images, labels in tqdm(test_loader, desc='Adaptive BPDA'):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # FIX 2: BPDA uses backward — no no_grad wrapping here
            adv_images = adaptive_attack.attack(images, self.model, labels)

            with torch.no_grad():
                with autocast(device_type=self.device):
                    outputs = self.model(adv_images.detach(),
                                        return_detailed=True)
                detected_mask = self._detect_from_scores(outputs['avg_anomaly_score'])
                preds     = outputs['logits'].float().argmax(dim=1)
                correct  += (preds == labels).sum().item()
                detected_count_total += detected_mask.sum().item()
                total    += labels.size(0)

            batch_count += 1
            if batch_count >= max_batches:
                break

        return {
            'robust_accuracy':              100.0 * correct / total,
            'attack_success_rate':          100.0 * (1 - correct / total),
            'detection_true_positive_rate': 100.0 * detected_count_total / total,
        }

    # ── full evaluation ───────────────────────────────────────────────────────
    def evaluate_all(self, test_loader, max_batches=20) -> dict:
        results = {}

        print('\n' + '=' * 60)
        print('AMSDN COMPREHENSIVE EVALUATION')
        print('=' * 60)

        threshold = self.calibrate_detection_threshold(test_loader, max_batches=20)
        print(f'Calibrated detection threshold: {threshold:.4f} (target FPR {self.target_fpr:.1f}%)')

        # Clean
        clean_results = self.evaluate_clean(test_loader)
        results['clean'] = clean_results
        print(f"\nClean Accuracy : {clean_results['accuracy']:.2f}%")
        print(f"FPR            : {clean_results['false_positive_rate']:.2f}%")
        print(f"Throughput     : {clean_results['throughput_img_per_sec']:.1f} img/sec")

        # Per-attack
        for attack_name, attack in self.attacks.items():
            attack_results = self.evaluate_attack(
                test_loader, attack_name, attack, max_batches)
            results[attack_name] = attack_results
            print(f"\n{attack_name}")
            print(f"  Robust Acc   : {attack_results['robust_accuracy']:.2f}%")
            print(f"  ASR          : {attack_results['attack_success_rate']:.2f}%")
            print(f"  Detection TPR: {attack_results['detection_true_positive_rate']:.2f}%")

        # Adaptive
        adaptive_results = self.evaluate_adaptive_attack(test_loader)
        results['Adaptive-BPDA'] = adaptive_results
        print('\nAdaptive BPDA')
        print(f"  Robust Acc   : {adaptive_results['robust_accuracy']:.2f}%")
        print(f"  ASR          : {adaptive_results['attack_success_rate']:.2f}%")
        print(f"  Detection TPR: {adaptive_results['detection_true_positive_rate']:.2f}%")

        return results

    def save_results(self, results,
                     path='./results/evaluation_results.json'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f'\n✓ Results saved to {path}')


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # FIX 4: keep device as plain string throughout
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    if device == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    data_module = CIFAR10DataModule(batch_size=64, num_workers=2)
    _, test_loader = data_module.get_loaders()

    model = AMSDN(num_classes=10, pretrained=False)

    # FIX 1: check fast_fgsm checkpoint first, then finetuned, then adversarial
    checkpoint_candidates = [
        './checkpoints/fast_robust_low_resource/amsdn_fast_robust_best.pth',  # fast robust low-resource
        './checkpoints/fast_robust/amsdn_fast_robust_best.pth',               # fast robust tuning
        './checkpoints/fast_fgsm/amsdn_fast_fgsm_best.pth',                   # older fast path
        './checkpoints/finetuned/amsdn_finetuned_best.pth',                   # multi-attack finetuned
        './checkpoints/adversarial/amsdn_best.pth',                           # base adversarial
    ]

    checkpoint_path = None
    for path in checkpoint_candidates:
        if os.path.exists(path):
            checkpoint_path = path
            break

    if checkpoint_path:
        print(f'Loading checkpoint: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path,
                                map_location=device,
                                weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Print what was saved with this checkpoint
        if 'results' in checkpoint:
            print(f"Checkpoint epoch : {checkpoint.get('epoch', 'N/A')}")
            print("Saved results    :")
            for k, v in checkpoint['results'].items():
                print(f"  {k:>8s}: {v:.2f}%")
    else:
        print('Warning: No trained model found. Evaluating random weights.')

    evaluator = AMSDNEvaluator(model, device=device)
    results   = evaluator.evaluate_all(test_loader, max_batches=20)
    evaluator.save_results(results)


if __name__ == '__main__':
    main()