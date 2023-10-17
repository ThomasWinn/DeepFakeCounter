import math

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

from helper import compute_mean_and_std

class DataLoader():
    def __init__(
            self, 
            file, 
            batch_size: int = 32,
            valid_size: float = 0.2,
            num_workers: int = 0
        ):
        self.stdev_file = file
        self.batch_size = batch_size
        self.valid_size = valid_size
        self.num_workers = num_workers

    def _train_transform(self, mean, std):
        return transforms.Compose(
            transforms.ToTensor()
        )

    def _valid_transform(self, mean, std):
        return transforms.Compose(
            transforms.ToTensor()
        )

    def _test_transform(self, mean, std):
        return transforms.Compose(
            transforms.ToTensor()
        )

    def _train_imagefolder(self, train_path):
        return ImageFolder(train_path)

    def _test_imagefolder(self, test_path):
        return ImageFolder(test_path)

    def get_train_val_dataloader(self):
        mean, std = compute_mean_and_std()
        train_data = ImageFolder(
            '../../datasets/train',
            self._train_transform(mean, std)
        )
        valid_data = ImageFolder(
            '../../dataset/train',
            self._valid_transform(mean, std)
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
        mean, std = compute_mean_and_std()
        test_data = ImageFolder(
            '../../dataset/train',
            self._valid_transform(mean, std)
        )
        return DataLoader(
            test_data,
            batch_size=self.batch_size,
            sampler=None,
            num_workers=self.num_workers,
            shuffle=False
        )
    
if __name__ == "__main__":
    d = DataLoader()
    train, val = d.get_train_val_dataloader()
    test = d.get_test_dataloader()