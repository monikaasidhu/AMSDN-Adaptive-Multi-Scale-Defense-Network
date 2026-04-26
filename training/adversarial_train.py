"""
Adversarial Training for AMSDN
Joint detection + purification + classification training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.amsdn import AMSDN
from data.cifar10 import CIFAR10DataModule


def load_ssrt_weights_if_available(model, checkpoint_path, device):
    """
    Load the AMSDN submodule weights from an SSRT checkpoint if present.
    This keeps SSRT as an optional warm-start instead of an isolated stage.
    """
    if not os.path.exists(checkpoint_path):
        return False

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", {})

    amsdn_state = {}
    for key, value in state_dict.items():
        if key.startswith("amsdn."):
            amsdn_state[key[len("amsdn."):]] = value

    if not amsdn_state:
        return False

    missing, unexpected = model.load_state_dict(amsdn_state, strict=False)
    print(f"Loaded SSRT warm-start from {checkpoint_path}")
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
    return True


# ============================================================
# PGD ATTACK (Strong + Adaptive Option)
# ============================================================

class PGDAttack:
    def __init__(self, epsilon=8/255, alpha=2/255, num_steps=10, adaptive=False, adaptive_lambda=0.5):
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.adaptive = adaptive
        self.adaptive_lambda = adaptive_lambda
        self.ce = nn.CrossEntropyLoss()

    def generate(self, model, images, labels):
        was_training = model.training
        model.eval()

        # Random start inside epsilon-ball
        images_adv = images + torch.empty_like(images).uniform_(-self.epsilon, self.epsilon)
        images_adv = torch.clamp(images_adv, -2, 2)

        for _ in range(self.num_steps):
            images_adv.requires_grad_(True)

            outputs = model(images_adv, return_detailed=True)
            logits = outputs['logits']

            ce_loss = self.ce(logits, labels)

            if self.adaptive:
                anomaly = outputs['avg_anomaly_score']
                loss = ce_loss - self.adaptive_lambda * anomaly.mean()
            else:
                loss = ce_loss

            grad = torch.autograd.grad(loss, images_adv, retain_graph=False, create_graph=False)[0]

            images_adv = images_adv.detach() + self.alpha * grad.sign()

            delta = torch.clamp(images_adv - images, -self.epsilon, self.epsilon)
            images_adv = torch.clamp(images + delta, -2, 2)

        model.train(was_training)
        return images_adv.detach()


class RandomPatchAttack:
    """Cheap localized corruption used to diversify detector positives."""

    def __init__(self, patch_size=4, epsilon=0.3):
        self.patch_size = patch_size
        self.epsilon = epsilon

    def generate(self, images):
        adv_images = images.clone()
        _, _, height, width = adv_images.shape

        for idx in range(adv_images.size(0)):
            top = torch.randint(0, height - self.patch_size + 1, (1,),
                                device=adv_images.device).item()
            left = torch.randint(0, width - self.patch_size + 1, (1,),
                                 device=adv_images.device).item()
            patch = torch.empty(
                3, self.patch_size, self.patch_size, device=adv_images.device
            ).uniform_(-self.epsilon, self.epsilon)
            adv_images[idx, :, top:top + self.patch_size,
                       left:left + self.patch_size] = patch

        return adv_images.clamp(-2, 2)


class RandomSparseAttack:
    """Cheap sparse corruption used to diversify detector positives."""

    def __init__(self, num_pixels=5, epsilon=0.5):
        self.num_pixels = num_pixels
        self.epsilon = epsilon

    def generate(self, images):
        adv_images = images.clone()
        batch_size, channels, height, width = adv_images.shape

        for batch_idx in range(batch_size):
            xs = torch.randint(0, height, (self.num_pixels,),
                               device=adv_images.device)
            ys = torch.randint(0, width, (self.num_pixels,),
                               device=adv_images.device)
            values = torch.empty(
                self.num_pixels, channels, device=adv_images.device
            ).uniform_(-self.epsilon, self.epsilon)
            adv_images[batch_idx, :, xs, ys] = values.transpose(0, 1)

        return adv_images.clamp(-2, 2)


# ============================================================
# TRAINER
# ============================================================

class AdversarialTrainer:

    def __init__(self, model, device='cuda', lr=1e-4, save_dir='./checkpoints',
                 use_amp=True, detection_weight=1.0, label_smoothing=0.0):
        self.model = model.to(device)
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.attack = PGDAttack()
        self.patch_attack = RandomPatchAttack()
        self.sparse_attack = RandomSparseAttack()
        self.detection_weight = detection_weight
        self.label_smoothing = label_smoothing

        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = None

        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Train the detector in logit space to avoid sigmoid saturation.
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.use_amp = use_amp and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        self.writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs'))

    # --------------------------------------------------------

    def compute_loss(self, outputs, labels, is_adv_gt):
        logits = outputs['logits']
        anomaly_logits = outputs['avg_anomaly_logit']

        cls_loss = self.ce_loss(logits, labels)

        det_loss = self.bce_loss(
            anomaly_logits.view(-1),
            is_adv_gt.float()
        )

        total_loss = cls_loss + self.detection_weight * det_loss
        return total_loss, cls_loss, det_loss

    def generate_mixed_adversarial_batch(self, images, labels):
        """
        Generate a diverse adversarial-positive batch while keeping the
        existing workflow unchanged.
        """
        batch_size = images.size(0)
        if batch_size == 0:
            return images

        # Deterministic partitioning avoids changing the outer workflow.
        pgd_end = max(1, batch_size // 2)
        patch_end = min(batch_size, pgd_end + max(1, batch_size // 4))

        adv_parts = []
        if pgd_end > 0:
            adv_parts.append(self.attack.generate(
                self.model, images[:pgd_end], labels[:pgd_end]
            ))
        if patch_end > pgd_end:
            adv_parts.append(self.patch_attack.generate(images[pgd_end:patch_end]))
        if batch_size > patch_end:
            adv_parts.append(self.sparse_attack.generate(images[patch_end:]))

        return torch.cat(adv_parts, dim=0)

    # --------------------------------------------------------

    def train_epoch(self, train_loader, epoch, num_epochs):
        self.model.train()

        total_loss = 0
        total_cls = 0
        total_det = 0

        correct_clean = 0
        correct_adv = 0
        clean_count = 0
        adv_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, (images, labels) in enumerate(pbar):

            images = images.to(self.device)
            labels = labels.to(self.device)
            batch_size = images.size(0)

            split = batch_size // 2

            clean_images = images[:split]
            clean_labels = labels[:split]

            adv_images = self.generate_mixed_adversarial_batch(
                images[split:], labels[split:]
            )
            adv_labels = labels[split:]

            combined_images = torch.cat([clean_images, adv_images], dim=0)
            combined_labels = torch.cat([clean_labels, adv_labels], dim=0)

            is_adv = torch.cat([
                torch.zeros(split, device=self.device),
                torch.ones(batch_size - split, device=self.device)
            ])

            # Shuffle batch
            perm = torch.randperm(batch_size, device=self.device)
            combined_images = combined_images[perm]
            combined_labels = combined_labels[perm]
            is_adv = is_adv[perm]

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(combined_images, return_detailed=True)
                    loss, cls_loss, det_loss = self.compute_loss(outputs, combined_labels, is_adv)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(combined_images, return_detailed=True)
                loss, cls_loss, det_loss = self.compute_loss(outputs, combined_labels, is_adv)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            preds = outputs['logits'].argmax(dim=1)

            clean_mask = (is_adv == 0)
            adv_mask = (is_adv == 1)

            correct_clean += (preds[clean_mask] == combined_labels[clean_mask]).sum().item()
            correct_adv += (preds[adv_mask] == combined_labels[adv_mask]).sum().item()

            clean_count += clean_mask.sum().item()
            adv_count += adv_mask.sum().item()

            total_loss += loss.item()
            total_cls += cls_loss.item()
            total_det += det_loss.item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "clean_acc": f"{100*correct_clean/max(1,clean_count):.1f}%",
                "adv_acc": f"{100*correct_adv/max(1,adv_count):.1f}%"
            })

        avg_loss = total_loss / len(train_loader)
        clean_acc = 100 * correct_clean / max(1, clean_count)
        adv_acc = 100 * correct_adv / max(1, adv_count)

        return avg_loss, clean_acc, adv_acc

    # --------------------------------------------------------

    def evaluate(self, test_loader):
        self.model.eval()

        correct_clean = 0
        correct_adv = 0
        detected_clean = 0
        detected_adv = 0
        total = 0

        attack = PGDAttack(num_steps=10)

        for images, labels in tqdm(test_loader, desc="Evaluating"):

            images = images.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                outputs_clean = self.model(images, return_detailed=True)
                preds_clean = outputs_clean['logits'].argmax(dim=1)

                correct_clean += (preds_clean == labels).sum().item()
                detected_clean += (outputs_clean['is_adversarial']).sum().item()

            adv_images = attack.generate(self.model, images, labels)

            with torch.no_grad():
                outputs_adv = self.model(adv_images, return_detailed=True)
                preds_adv = outputs_adv['logits'].argmax(dim=1)

                correct_adv += (preds_adv == labels).sum().item()
                detected_adv += (outputs_adv['is_adversarial']).sum().item()

            total += labels.size(0)

        clean_acc = 100 * correct_clean / total
        adv_acc = 100 * correct_adv / total

        tpr = 100 * detected_adv / total
        fpr = 100 * detected_clean / total

        return clean_acc, adv_acc, tpr, fpr

    # --------------------------------------------------------

    def train(self, train_loader, test_loader, num_epochs=20):

        best_detection_score = float('-inf')
        last_metrics = None
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=1e-6
        )

        for epoch in range(1, num_epochs+1):

            avg_loss, train_clean, train_adv = self.train_epoch(
                train_loader, epoch, num_epochs
            )
            last_metrics = {
                "epoch": epoch,
                "avg_loss": avg_loss,
                "train_clean": train_clean,
                "train_adv": train_adv,
            }

            if epoch % 5 == 0:
                clean_acc, adv_acc, tpr, fpr = self.evaluate(test_loader)
                last_metrics.update({
                    "test_clean": clean_acc,
                    "test_adv": adv_acc,
                    "tpr": tpr,
                    "fpr": fpr,
                })

                print(f"\nEpoch {epoch}")
                print(f"Train Clean: {train_clean:.2f}% | Train Adv: {train_adv:.2f}%")
                print(f"Test Clean: {clean_acc:.2f}% | Test Adv: {adv_acc:.2f}%")
                print(f"Detection TPR: {tpr:.2f}% | FPR: {fpr:.2f}%")

                detection_score = tpr - fpr
                selection_score = (
                    0.45 * adv_acc +
                    0.25 * clean_acc +
                    0.30 * detection_score
                )

                if selection_score > best_detection_score:
                    best_detection_score = selection_score
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "adv_accuracy": adv_acc,
                        "clean_accuracy": clean_acc,
                        "tpr": tpr,
                        "fpr": fpr,
                        "selection_score": selection_score,
                    }, os.path.join(self.save_dir, "amsdn_best.pth"))
                    print(f'Saved best checkpoint to {os.path.join(self.save_dir, "amsdn_best.pth")}')

            self.scheduler.step()

        final_checkpoint_path = os.path.join(self.save_dir, "amsdn_last.pth")
        torch.save({
            "epoch": num_epochs,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": last_metrics or {},
        }, final_checkpoint_path)
        print(f"Saved final checkpoint to {final_checkpoint_path}")

        self.writer.close()


# ============================================================
# MAIN
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_module = CIFAR10DataModule(batch_size=128, num_workers=2)
    train_loader, test_loader = data_module.get_loaders()

    model = AMSDN(num_classes=10, pretrained=True)
    load_ssrt_weights_if_available(model, "./checkpoints/ssrt/ssrt_best.pth", device)

    trainer = AdversarialTrainer(
        model,
        device=device,
        save_dir='./checkpoints/adversarial',
        detection_weight=1.0,
        label_smoothing=0.0
    )
    trainer.train(train_loader, test_loader, num_epochs=10)


if __name__ == "__main__":
    main()
