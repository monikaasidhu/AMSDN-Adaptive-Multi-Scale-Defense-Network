"""
Fast robust tuning for AMSDN.

This is a practical replacement for `finetune_attacks.py` when full
multi-attack fine-tuning is too slow or unstable to finish. Instead of
running expensive inner-loop attacks such as C&W, optimized patch attacks,
and iterative sparse attacks on every batch, this script uses:

1. FGSM for gradient-based robustness
2. Random patch corruption as a cheap proxy for localized attacks
3. Random sparse pixel corruption as a cheap proxy for few-pixel attacks

The core objective stays the same:
train AMSDN to remain accurate and detect anomalies under diverse,
attack-like perturbations.
"""

import argparse
import os
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.cifar10 import CIFAR10DataModule
from models.amsdn import AMSDN
from training.adversarial_train import PGDAttack


class FastAttackSuite:
    """Cheap perturbation generators used during fast robust tuning."""

    def __init__(self, epsilon=8 / 255, fgsm_alpha=8 / 255, patch_size=4,
                 patch_epsilon=0.3, num_pixels=5, pixel_epsilon=0.5):
        self.epsilon = epsilon
        self.fgsm_alpha = fgsm_alpha
        self.patch_size = patch_size
        self.patch_epsilon = patch_epsilon
        self.num_pixels = num_pixels
        self.pixel_epsilon = pixel_epsilon
        self.ce = nn.CrossEntropyLoss()

    def fgsm(self, model, images, labels):
        """Single-step gradient attack."""
        was_training = model.training
        model.eval()

        adv_images = images.detach().clone().requires_grad_(True)
        outputs = model(adv_images)
        loss = self.ce(outputs, labels)
        grad = torch.autograd.grad(
            loss, adv_images, retain_graph=False, create_graph=False
        )[0]

        adv_images = adv_images.detach() + self.fgsm_alpha * grad.sign()
        delta = torch.clamp(adv_images - images, -self.epsilon, self.epsilon)
        adv_images = torch.clamp(images + delta, -2, 2)

        model.train(was_training)
        return adv_images.detach()

    def random_patch(self, images):
        """Cheap localized corruption that approximates patch attacks."""
        adv_images = images.clone()
        _, _, height, width = adv_images.shape

        for idx in range(adv_images.size(0)):
            top = random.randint(0, height - self.patch_size)
            left = random.randint(0, width - self.patch_size)
            patch = torch.empty(
                3, self.patch_size, self.patch_size, device=adv_images.device
            ).uniform_(-self.patch_epsilon, self.patch_epsilon)
            adv_images[idx, :, top:top + self.patch_size,
                       left:left + self.patch_size] = patch

        return adv_images.clamp(-2, 2)

    def random_sparse(self, images):
        """Cheap sparse corruption that approximates few-pixel attacks."""
        adv_images = images.clone()
        batch_size, channels, height, width = adv_images.shape

        for batch_idx in range(batch_size):
            xs = torch.randint(0, height, (self.num_pixels,),
                               device=adv_images.device)
            ys = torch.randint(0, width, (self.num_pixels,),
                               device=adv_images.device)
            values = torch.empty(
                self.num_pixels, channels, device=adv_images.device
            ).uniform_(-self.pixel_epsilon, self.pixel_epsilon)
            adv_images[batch_idx, :, xs, ys] = values.transpose(0, 1)

        return adv_images.clamp(-2, 2)


class FastRobustTuner:
    """Fast replacement for expensive multi-attack fine-tuning."""

    def __init__(self, model, device='cuda', lr=5e-5,
                 save_dir='./checkpoints/fast_robust', eval_pgd_steps=10,
                 freeze_backbone_epochs=2, use_amp=True,
                 detection_weight=0.3, label_smoothing=0.0):
        self.model = model.to(device)
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.attack_suite = FastAttackSuite()
        self.eval_attack = PGDAttack(num_steps=eval_pgd_steps)
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=1e-5
        )
        self.scheduler = None
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.detection_weight = detection_weight
        self.use_amp = use_amp and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

    def set_backbone_trainable(self, trainable):
        for param in self.model.backbone.parameters():
            param.requires_grad = trainable

    def generate_batch(self, images, labels, epoch):
        """
        Use a simple curriculum:
        early epochs focus on FGSM only,
        later epochs mix in patch-like and sparse-like corruptions.
        """
        if epoch <= 2:
            mode = random.choice(['clean', 'fgsm'])
        elif epoch <= 4:
            mode = random.choice(['clean', 'fgsm', 'patch', 'sparse'])
        else:
            mode = random.choice(['fgsm', 'fgsm', 'patch', 'sparse', 'clean'])

        if mode == 'fgsm':
            return self.attack_suite.fgsm(self.model, images, labels), mode, 1.0
        if mode == 'patch':
            return self.attack_suite.random_patch(images), mode, 1.0
        if mode == 'sparse':
            return self.attack_suite.random_sparse(images), mode, 1.0
        return images, mode, 0.0

    def compute_loss(self, outputs, labels, is_adv):
        cls_loss = self.ce_loss(outputs['logits'], labels)
        det_loss = self.bce_loss(outputs['avg_anomaly_logit'].view(-1), is_adv)
        total_loss = cls_loss + self.detection_weight * det_loss
        return total_loss, cls_loss, det_loss

    def train_epoch(self, train_loader, epoch):
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0
        mix_counts = {'clean': 0, 'fgsm': 0, 'patch': 0, 'sparse': 0}

        pbar = tqdm(train_loader, desc=f'Fast tune epoch {epoch}')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            mixed_images, mode, is_adv_flag = self.generate_batch(
                images, labels, epoch
            )
            mix_counts[mode] += 1

            is_adv = torch.full(
                (labels.size(0),), float(is_adv_flag), device=self.device
            )

            self.optimizer.zero_grad()
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(mixed_images, return_detailed=True)
                    loss, _, _ = self.compute_loss(outputs, labels, is_adv)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(mixed_images, return_detailed=True)
                loss, _, _ = self.compute_loss(outputs, labels, is_adv)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            preds = outputs['logits'].argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / max(1, total):.1f}%',
                'mode': mode
            })

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / max(1, total)
        return avg_loss, accuracy, mix_counts

    def evaluate(self, test_loader, max_batches=20):
        self.model.eval()

        clean_correct = 0
        fgsm_correct = 0
        pgd_correct = 0
        patch_correct = 0
        sparse_correct = 0
        total = 0
        batches = 0

        for images, labels in tqdm(test_loader, desc='Fast tune eval'):
            images = images.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                clean_outputs = self.model(images)
                clean_correct += (clean_outputs.argmax(dim=1) == labels).sum().item()

            fgsm_images = self.attack_suite.fgsm(self.model, images, labels)
            patch_images = self.attack_suite.random_patch(images)
            sparse_images = self.attack_suite.random_sparse(images)
            pgd_images = self.eval_attack.generate(self.model, images, labels)

            with torch.no_grad():
                fgsm_correct += (self.model(fgsm_images).argmax(dim=1) == labels).sum().item()
                patch_correct += (self.model(patch_images).argmax(dim=1) == labels).sum().item()
                sparse_correct += (self.model(sparse_images).argmax(dim=1) == labels).sum().item()
                pgd_correct += (self.model(pgd_images).argmax(dim=1) == labels).sum().item()

            total += labels.size(0)
            batches += 1
            if batches >= max_batches:
                break

        return {
            'clean': 100.0 * clean_correct / max(1, total),
            'fgsm': 100.0 * fgsm_correct / max(1, total),
            'patch_proxy': 100.0 * patch_correct / max(1, total),
            'sparse_proxy': 100.0 * sparse_correct / max(1, total),
            'pgd_eval': 100.0 * pgd_correct / max(1, total),
        }

    def train(self, train_loader, test_loader, num_epochs=12):
        print('=' * 60)
        print('Starting Fast Robust Tuning')
        print('=' * 60)

        best_score = 0.0
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=1e-6
        )

        for epoch in range(1, num_epochs + 1):
            self.set_backbone_trainable(epoch > self.freeze_backbone_epochs)
            train_loss, train_acc, mix_counts = self.train_epoch(train_loader, epoch)

            if epoch % 3 == 0:
                results = self.evaluate(test_loader, max_batches=15)
                proxy_robust = (
                    0.40 * results['pgd_eval'] +
                    0.20 * results['fgsm'] +
                    0.20 * results['patch_proxy'] +
                    0.20 * results['sparse_proxy']
                )

                print(f'\nEpoch {epoch}/{num_epochs}')
                print(f'  Train Loss: {train_loss:.4f}')
                print(f'  Train Acc : {train_acc:.2f}%')
                print(f'  Mix       : {mix_counts}')
                for key, value in results.items():
                    print(f'  {key:12s}: {value:.2f}%')

                if proxy_robust > best_score:
                    best_score = proxy_robust
                    checkpoint_path = os.path.join(
                        self.save_dir, 'amsdn_fast_robust_best.pth'
                    )
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'results': results,
                    }, checkpoint_path)
                    print(f'  Saved best model (proxy robust: {best_score:.2f}%)')

            self.scheduler.step()

        print('\n' + '=' * 60)
        print('Fast robust tuning complete')
        print(f'Best proxy robust score: {best_score:.2f}%')
        print('=' * 60)


def build_config(low_resource=False):
    """Return a practical config for normal or low-resource runs."""
    if low_resource:
        return {
            'batch_size': 64,
            'num_workers': 2,
            'num_epochs': 6,
            'lr': 7e-5,
            'eval_batches': 6,
            'eval_every': 2,
            'eval_pgd_steps': 5,
            'freeze_backbone_epochs': 4,
            'save_dir': './checkpoints/fast_robust_low_resource',
        }

    return {
        'batch_size': 128,
        'num_workers': 2,
        'num_epochs': 12,
        'lr': 5e-5,
        'eval_batches': 15,
        'eval_every': 3,
        'eval_pgd_steps': 10,
        'freeze_backbone_epochs': 2,
        'save_dir': './checkpoints/fast_robust',
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Fast robust tuning for AMSDN')
    parser.add_argument(
        '--low-resource',
        action='store_true',
        help='Use a lighter preset with fewer epochs, smaller batches, and cheaper evaluation.'
    )
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--eval-batches', type=int, default=None)
    parser.add_argument('--eval-every', type=int, default=None)
    parser.add_argument('--eval-pgd-steps', type=int, default=None)
    parser.add_argument('--freeze-backbone-epochs', type=int, default=None)
    parser.add_argument('--save-dir', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = build_config(low_resource=args.low_resource)

    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.epochs is not None:
        config['num_epochs'] = args.epochs
    if args.lr is not None:
        config['lr'] = args.lr
    if args.eval_batches is not None:
        config['eval_batches'] = args.eval_batches
    if args.eval_every is not None:
        config['eval_every'] = args.eval_every
    if args.eval_pgd_steps is not None:
        config['eval_pgd_steps'] = args.eval_pgd_steps
    if args.freeze_backbone_epochs is not None:
        config['freeze_backbone_epochs'] = args.freeze_backbone_epochs
    if args.save_dir is not None:
        config['save_dir'] = args.save_dir

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Low-resource mode: {args.low_resource}')
    print(f'Config: {config}')

    data_module = CIFAR10DataModule(
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    train_loader, test_loader = data_module.get_loaders()

    model = AMSDN(num_classes=10, pretrained=False)
    checkpoint_candidates = [
        './checkpoints/adversarial/amsdn_best.pth',
        './checkpoints/adversarial/amsdn_last.pth',
        './checkpoints/amsdn_best.pth',
        './checkpoints/amsdn_last.pth',
    ]
    checkpoint_path = None
    for path in checkpoint_candidates:
        if os.path.exists(path):
            checkpoint_path = path
            break

    if checkpoint_path:
        print(f'Loading model from {checkpoint_path}...')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print('Warning: No adversarial checkpoint found. Starting from scratch.')

    trainer = FastRobustTuner(
        model=model,
        device=device,
        lr=config['lr'],
        save_dir=config['save_dir'],
        eval_pgd_steps=config['eval_pgd_steps'],
        freeze_backbone_epochs=config['freeze_backbone_epochs'],
        use_amp=True,
        detection_weight=0.3,
        label_smoothing=0.0
    )
    original_evaluate = trainer.evaluate

    def configured_evaluate(loader, max_batches=None):
        return original_evaluate(
            loader,
            max_batches=config['eval_batches'] if max_batches is None else max_batches
        )

    trainer.evaluate = configured_evaluate

    original_train = trainer.train

    def configured_train(train_loader_arg, test_loader_arg, num_epochs=None):
        print(f'Evaluating every {config["eval_every"]} epochs')
        best_score = 0.0
        trainer.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer, T_max=config['num_epochs'], eta_min=1e-6
        )

        print('=' * 60)
        print('Starting Fast Robust Tuning')
        print('=' * 60)

        for epoch in range(1, config['num_epochs'] + 1):
            trainer.set_backbone_trainable(epoch > trainer.freeze_backbone_epochs)
            train_loss, train_acc, mix_counts = trainer.train_epoch(train_loader_arg, epoch)

            if epoch % config['eval_every'] == 0:
                results = trainer.evaluate(test_loader_arg)
                proxy_robust = (
                    results['fgsm'] +
                    results['patch_proxy'] +
                    results['sparse_proxy'] +
                    results['pgd_eval']
                ) / 4.0

                print(f'\nEpoch {epoch}/{config["num_epochs"]}')
                print(f'  Train Loss: {train_loss:.4f}')
                print(f'  Train Acc : {train_acc:.2f}%')
                print(f'  Mix       : {mix_counts}')
                for key, value in results.items():
                    print(f'  {key:12s}: {value:.2f}%')

                if proxy_robust > best_score:
                    best_score = proxy_robust
                    checkpoint_path = os.path.join(
                        trainer.save_dir, 'amsdn_fast_robust_best.pth'
                    )
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': trainer.model.state_dict(),
                        'results': results,
                    }, checkpoint_path)
                    print(f'  Saved best model (proxy robust: {best_score:.2f}%)')

            trainer.scheduler.step()

        print('\n' + '=' * 60)
        print('Fast robust tuning complete')
        print(f'Best proxy robust score: {best_score:.2f}%')
        print('=' * 60)

    trainer.train = configured_train
    trainer.train(train_loader, test_loader, num_epochs=config['num_epochs'])


if __name__ == '__main__':
    main()
