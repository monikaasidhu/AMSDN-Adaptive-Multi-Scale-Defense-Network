"""
Fine-tuning with diverse attacks (PGD, C&W, Patch)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.amsdn import AMSDN
from data.cifar10 import CIFAR10DataModule
from attacks.patch_attacks import AdversarialPatch
from attacks.pixel_attacks import FewPixelAttack


class CWAttack:
    """Carlini-Wagner L2 attack (simplified)"""
    
    def __init__(self, c=0.1, kappa=0, num_steps=100, lr=0.01):
        self.c = c
        self.kappa = kappa
        self.num_steps = num_steps
        self.lr = lr
    
    def generate(self, model, images, labels):
        """Generate C&W adversarial examples"""
        batch_size = images.size(0)
        
        # Initialize perturbation
        delta = torch.zeros_like(images, requires_grad=True)
        optimizer = optim.Adam([delta], lr=self.lr)
        
        for step in range(self.num_steps):
            adv_images = images + delta
            adv_images = torch.clamp(adv_images, -2, 2)
            
            logits = model(adv_images)
            
            # C&W loss
            real = logits.gather(1, labels.unsqueeze(1)).squeeze()
            other = logits.clone()
            other.scatter_(1, labels.unsqueeze(1), -float('inf'))
            other_max = other.max(dim=1)[0]
            
            f_loss = torch.clamp(real - other_max + self.kappa, min=0).sum()
            l2_loss = torch.norm(delta.view(batch_size, -1), p=2, dim=1).sum()
            
            loss = l2_loss + self.c * f_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Early stopping if misclassified
            if step % 20 == 0:
                with torch.no_grad():
                    preds = model(images + delta).argmax(dim=1)
                    if (preds != labels).all():
                        break
        
        return (images + delta.detach()).clamp(-2, 2)


class MultiAttackFineTuner:
    """Fine-tune AMSDN with multiple attack types"""
    
    def __init__(self, model, device='cuda', lr=5e-5, save_dir='./checkpoints'):
        self.model = model.to(device)
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Multiple attacks
        self.attacks = {
            'pgd': self.pgd_attack,
            'cw': CWAttack(c=0.1, num_steps=50),
            'patch': AdversarialPatch(patch_size=4, epsilon=0.3),
            'pixel': FewPixelAttack(num_pixels=5, epsilon=0.5)
        }
        
        # Optimizer (lower learning rate for fine-tuning)
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        
        # Loss
        self.ce_loss = nn.CrossEntropyLoss()
        
    def pgd_attack(self, images, labels):
        """PGD attack"""
        epsilon = 8/255
        alpha = 2/255
        num_steps = 20
        
        images_adv = images.clone()
        for _ in range(num_steps):
            images_adv.requires_grad = True
            outputs = self.model(images_adv)
            loss = self.ce_loss(outputs, labels)
            grad = torch.autograd.grad(loss, images_adv)[0]
            images_adv = images_adv.detach() + alpha * grad.sign()
            delta = torch.clamp(images_adv - images, -epsilon, epsilon)
            images_adv = torch.clamp(images + delta, -2, 2)
        
        return images_adv
    
    def generate_mixed_attacks(self, images, labels):
        """Generate adversarial examples from random attack"""
        attack_name = list(self.attacks.keys())[
            torch.randint(0, len(self.attacks), (1,)).item()
        ]
        
        if attack_name == 'pgd':
            return self.attacks['pgd'](images, labels), attack_name
        elif attack_name == 'cw':
            return self.attacks['cw'].generate(self.model, images, labels), attack_name
        elif attack_name == 'patch':
            return self.attacks['patch'].apply(images, self.model, labels), attack_name
        else:  # pixel
            return self.attacks['pixel'].attack(images, self.model, labels), attack_name
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch with mixed attacks"""
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        attack_counts = {k: 0 for k in self.attacks.keys()}
        
        pbar = tqdm(train_loader, desc=f'Fine-tune Epoch {epoch}')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Generate adversarial examples from random attack
            adv_images, attack_name = self.generate_mixed_attacks(images, labels)
            attack_counts[attack_name] += 1
            
            # Forward pass
            outputs = self.model(adv_images)
            loss = self.ce_loss(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            # Metrics
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*correct/total:.1f}%',
                'attack': attack_name
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        print(f"\nAttack distribution: {attack_counts}")
        
        return avg_loss, accuracy
    
    def evaluate(self, test_loader):
        """Evaluate on multiple attacks"""
        self.model.eval()
        
        results = {}
        
        # Clean accuracy
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Clean eval'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        results['clean'] = 100 * correct / total
        
        # Test each attack
        for attack_name in self.attacks.keys():
            correct = 0
            total = 0
            
            for images, labels in tqdm(test_loader, desc=f'{attack_name} eval'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                if attack_name == 'pgd':
                    adv_images = self.attacks['pgd'](images, labels)
                elif attack_name == 'cw':
                    adv_images = self.attacks['cw'].generate(self.model, images, labels)
                elif attack_name == 'patch':
                    adv_images = self.attacks['patch'].apply(images, self.model, labels)
                else:
                    adv_images = self.attacks['pixel'].attack(images, self.model, labels)
                
                with torch.no_grad():
                    outputs = self.model(adv_images)
                    preds = outputs.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            
            results[attack_name] = 100 * correct / total
        
        return results
    
    def finetune(self, train_loader, test_loader, num_epochs=20):
        """Fine-tuning loop"""
        print("=" * 60)
        print("Starting Multi-Attack Fine-tuning")
        print("=" * 60)
        
        best_avg_acc = 0
        
        for epoch in range(1, num_epochs + 1):
            # Train
            avg_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Evaluate every 5 epochs
            if epoch % 5 == 0:
                results = self.evaluate(test_loader)
                
                print(f"\nEpoch {epoch}/{num_epochs}:")
                print(f"  Train - Loss: {avg_loss:.4f}, Acc: {train_acc:.2f}%")
                print(f"  Test Results:")
                for attack_name, acc in results.items():
                    print(f"    {attack_name}: {acc:.2f}%")
                
                # Average robust accuracy
                avg_robust = sum([v for k, v in results.items() if k != 'clean']) / (len(results) - 1)
                
                if avg_robust > best_avg_acc:
                    best_avg_acc = avg_robust
                    checkpoint_path = os.path.join(self.save_dir, 'amsdn_finetuned_best.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'results': results,
                    }, checkpoint_path)
                    print(f"  ✓ Saved best model (avg robust: {best_avg_acc:.2f}%)")
        
        print("\n" + "=" * 60)
        print("Fine-tuning Complete!")
        print(f"Best Average Robust Accuracy: {best_avg_acc:.2f}%")
        print("=" * 60)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    data_module = CIFAR10DataModule(batch_size=64, num_workers=2)
    train_loader, test_loader = data_module.get_loaders()
    
    # Load adversarially trained model
    model = AMSDN(num_classes=10, pretrained=False)
    checkpoint_path = './checkpoints/adversarial/amsdn_best.pth'
    
    if os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Warning: No pretrained model found. Starting from scratch.")
    
    # Fine-tuner
    finetuner = MultiAttackFineTuner(
        model=model,
        device=device,
        lr=5e-5,
        save_dir='./checkpoints/finetuned'
    )
    
    # Fine-tune
    finetuner.finetune(train_loader, test_loader, num_epochs=20)


if __name__ == "__main__":
    main()