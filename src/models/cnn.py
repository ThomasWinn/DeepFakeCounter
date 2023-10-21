from typing import Any
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import torch.nn.functional as F

class CIFAKE_CNN(nn.Module):
    """This is the proposed architecture for lowest loss with 92.93% accuracy
    
    The highest accuracy was using 2 conv layers of 128 filters scoring 92.98% accuracy

    Args:
        nn (Module): Inherit the base Module class
    """
    def __init__(self) -> None:
        super().__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # 32 x 32 x 32
            nn.MaxPool2d(2, 2), # 32 x 16 x 16
            nn.ReLU(),
            
            nn.Conv2d(32, 32, kernel_size=3, padding=1), # 32 x 16 x 16
            nn.MaxPool2d(2, 2), # 32 x 8 x 8
            nn.ReLU(),
        )
        
        self.flatten = nn.Flatten()
        
        self.head = nn.Sequential(
            nn.Linear(32 * 8 * 8, 64), # 2048 x 64
            nn.ReLU(),
            
            nn.Linear(64, 1), # 64 x 1
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.head(x)
        return x
    

class pl_CIFAKE_CNN(pl.LightningModule):
    """This is the proposed architecture for lowest loss with 92.93% accuracy
    
    The highest accuracy was using 2 conv layers of 128 filters scoring 92.98% accuracy

    Args:
        nn (Module): Inherit the base Module class
    """
    def __init__(self) -> None:
        super().__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # 32 x 32 x 32
            nn.MaxPool2d(2, 2), # 32 x 16 x 16
            nn.ReLU(),
            
            nn.Conv2d(32, 32, kernel_size=3, padding=1), # 32 x 16 x 16
            nn.MaxPool2d(2, 2), # 32 x 8 x 8
            nn.ReLU(),
        )
        
        self.flatten = nn.Flatten()
        
        self.head = nn.Sequential(
            nn.Linear(32 * 8 * 8, 64), # 2048 x 64
            nn.ReLU(),
            
            nn.Linear(64, 1), # 64 x 1
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.head(x)
        return x
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        y = y.float()
        y_hat = self.forward(x)
        y_hat = y_hat.squeeze()
        loss = F.cross_entropy(y_hat, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()
        y_hat = self.forward(x)
        y_hat = y_hat.squeeze()
        loss = F.cross_entropy(y_hat, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        y = y.float()
        y_hat = self.forward(x)
        y_hat = y_hat.squeeze()
        test_loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", test_loss)
    
    # TODO: Check if we can send in lr and weight decay to this function  or other vars
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=0)
        return optimizer