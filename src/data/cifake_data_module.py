import lightning.pytorch as pl
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torchvision.datasets as datasets

from .helper import compute_mean_and_std

class CIFAKEDataModule(pl.LightningDataModule):
    def __init__(self, cache_file, data_dir, batch_size, num_workers, valid_size):
        self.cache_file = cache_file
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
    
    def prepare_data(self) -> None:
        # download, split, etc...
        self.mean, self.std = compute_mean_and_std(self.cache_file)
    
    def setup(self, stage: str) -> None:
        # if stage == 'train'
        pass
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return super().train_dataloader()
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return super().val_dataloader()
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return super().test_dataloader()
    
    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return super().predict_dataloader()