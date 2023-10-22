from typing import Any, List
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torchmetrics import Metric


class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, targets):
        y_hat = (preds > 0.5).float()
        assert y_hat.shape == targets.shape
        
        self.correct += torch.sum(y_hat == targets)
        self.total += targets.numel()
        
    def compute(self):
        return self.correct.float() / self.total.float()

class CIFAKE_CNN(pl.LightningModule):
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
        
        self.cross_entropy = nn.CrossEntropyLoss()
        self.my_accuracy = MyAccuracy()
        self.accuracy = torchmetrics.Accuracy(task='binary', num_classes=2)
        self.f1_score = torchmetrics.F1Score(task='binary', num_classes=2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.head(x)
        return x
    
    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx)
        accuracy = self.my_accuracy(y_hat, y)
        f1_score = self.f1_score(y_hat, y)
        # slow; create an training_end function to calculate at end
        self.log_dict({'train_loss': loss, 'train_accuracy': accuracy, 'train_f1_score': f1_score},
                      on_step=False, on_epoch=True, prog_bar=True)
        # self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx)
        self.log("valid_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx)
        self.log("test_loss", loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        # x, y = batch
        # x = x.reshape(x.size(0), -1)
        # y_hat = self.forward(x)
        x, y = batch
        y = y.float()
        y_hat = self.forward(x)
        y_hat = y_hat.squeeze()
        return ((y_hat > 0.5).float()) == y
        # accuracy = (y_hat.round() == y).float().mean()
        # return float(accuracy)
        
        
    def _common_step(self, batch, batch_idx):
        # x, y = batch
        # x = x.reshape(x.size(0), -1)
        # y_hat = self.forward(x)
        x, y = batch
        y = y.float()
        y_hat = self.forward(x)
        y_hat = y_hat.squeeze()
        loss = self.cross_entropy(y_hat, y)
        return loss, y_hat, y
    
    def configure_optimizers(self):
        optimizer1 = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=0)
        return optimizer1
        # scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1,)
        # return (
        #     {
        #         "optimizer": optimizer1,
        #         "lr_scheduler": scheduler1
        #     }
        # )