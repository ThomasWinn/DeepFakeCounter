import math

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder


from .helper import compute_mean_and_std

class CIFAKEDataModule(pl.LightningDataModule):
    def __init__(self, cache_file, data_dir, batch_size, num_workers, valid_size):
        super().__init__()
        self.cache_file = cache_file
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        
    def prepare_data(self):
        # download, split, etc...
        self.mean, self.std = compute_mean_and_std(self.cache_file)
    
    def setup(self, stage: str) -> None:
        self.train_data = ImageFolder(
            '../dataset/train',
            transform=transforms.Compose([
                # transforms.Resize((32, 32)),
                # transforms.RandomAffine(scale=(0.9, 1.1), translate=(0.1, 0.1), degrees=10),
                # transforms.ColorJitter(brightness=(0.5,1.5),contrast=(1),saturation=(0.5,1.5),hue=(-0.1,0.1)),
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomRotation(90),
                transforms.ToTensor(),
                # transforms.Normalize(self.mean, self.std)
            ])
        )
        self.valid_data = ImageFolder(
            '../dataset/train',
            transform=transforms.Compose([
                # transforms.Resize((32, 32)),
                transforms.ToTensor(),
                # transforms.Normalize(self.mean, self.std)
            ])
        )
        
        # Experiment with SubsetRandomSampler() and random_split
        # SubsetRandomSampler - split on the indicie and assign randomly from that distribution
        # random_split - no control over split whole dataset is divided randomly

        # randomize indices, split dataset at 80:20, sample elements within randomized indices at random
        total_img = len(self.train_data)
        idxs = torch.randperm(total_img)

        split = int(math.ceil(self.valid_size * total_img))
        train_idx, valid_idx = idxs[split:], idxs[:split]

        self.train_sampler = SubsetRandomSampler(train_idx)
        self.valid_sampler = SubsetRandomSampler(valid_idx)
        
        self.test_data = ImageFolder(
            '../dataset/test',
            transform=transforms.Compose([
                # transforms.Resize((32, 32)),
                transforms.ToTensor(),
                # transforms.Normalize(self.mean, self.std)
            ])
        )
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
        )
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.valid_data,
            batch_size=self.batch_size,
            sampler=self.valid_sampler,
            num_workers=self.num_workers,
            shuffle=False
        )
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
    
    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return super().predict_dataloader()