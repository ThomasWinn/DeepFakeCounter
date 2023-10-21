import os
import math

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

from .helper import compute_mean_and_std

class Data_DataLoader():
    def __init__(
            self, 
            cache_file: str = 'mean_and_std.pt',
            batch_size: int = 32,
            valid_size: float = 0.2,
            num_workers: int = 0
        ):
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.valid_size = valid_size
        self.num_workers = num_workers

    def _train_transform(self, mean, std):
        return transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.RandomAffine(scale=(0.9, 1.1), translate=(0.1, 0.1), degrees=10),
                transforms.ColorJitter(brightness=(0.5,1.5),contrast=(1),saturation=(0.5,1.5),hue=(-0.1,0.1)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(90),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]
        )

    def _valid_transform(self, mean, std):
        return transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]
        )

    def _test_transform(self, mean, std):
        return transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]
        )

    def _train_imagefolder(self, train_path):
        return ImageFolder(train_path)

    def _test_imagefolder(self, test_path):
        return ImageFolder(test_path)

    def get_train_val_dataloader(self):
        mean, std = compute_mean_and_std(self.cache_file)
        folder = '../dataset/train' # ../../dataset/train when running in here

        train_data = ImageFolder(
            folder,
            transform=self._train_transform(mean, std)
        )
        valid_data = ImageFolder(
            folder,
            transform=self._valid_transform(mean, std)
        )

        # Experiment with SubsetRandomSampler() and random_split
        # SubsetRandomSampler - split on the indicie and assign randomly from that distribution
        # random_split - no control over split whole dataset is divided randomly

        # randomize indices, split dataset at 80:20, sample elements within randomized indices at random
        total_img = len(train_data)
        idxs = torch.randperm(total_img)

        split = int(math.ceil(self.valid_size * total_img))
        train_idx, valid_idx = idxs[split:], idxs[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        return DataLoader(
            train_data,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=self.num_workers
        ), DataLoader(
            valid_data,
            batch_size=self.batch_size,
            sampler=valid_sampler,
            num_workers=self.num_workers,
            shuffle=False
        )
    
    def get_test_dataloader(self):
        mean, std = compute_mean_and_std(self.cache_file)
        test_data = ImageFolder(
            '../dataset/test', # ../../dataset/test wheh running in here
            self._test_transform(mean, std)
        )
        return DataLoader(
            test_data,
            batch_size=self.batch_size,
            sampler=None,
            num_workers=self.num_workers,
            shuffle=False
        )
    
if __name__ == "__main__":
    d = Data_DataLoader()
    train, val = d.get_train_val_dataloader()
    test = d.get_test_dataloader()