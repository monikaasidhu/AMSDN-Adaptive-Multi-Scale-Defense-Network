import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np


class CIFAR10DataModule:
    """CIFAR-10 data loader with augmentations for AMSDN"""
    
    def __init__(self, data_dir='./data', batch_size=128, num_workers=2):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Standard normalization for CIFAR-10
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2470, 0.2435, 0.2616]
        
        # Training augmentation
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        
        # Test transform (no augmentation)
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        
    def get_loaders(self):
        """Returns train and test DataLoaders"""
        train_dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.train_transform
        )
        
        test_dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.test_transform
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, test_loader


class SSRTDataset(Dataset):
    """Self-Supervised dataset for SSRT pretraining"""
    
    def __init__(self, base_dataset, mask_ratio=0.25):
        self.base_dataset = base_dataset
        self.mask_ratio = mask_ratio
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        
        # Create masked version
        mask = torch.rand_like(img) > self.mask_ratio
        masked_img = img * mask.float()
        
        return masked_img, img, label  # masked, original, label


def get_ssrt_loader(data_dir='./data', batch_size=128, num_workers=2):
    """Get loader for self-supervised pretraining"""
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    base_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    ssrt_dataset = SSRTDataset(base_dataset)
    
    loader = DataLoader(
        ssrt_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader