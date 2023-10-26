from typing import Any, List, Union
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision

from .metrics import MyAccuracy


class CIFAKE_CNN(pl.LightningModule):
    """This is the proposed architecture for lowest loss with 92.93% accuracy

    The highest accuracy was using 2 conv layers of 128 filters scoring 92.98% accuracy

    Args:
        nn (Module): Inherit the base Module class
    """

    def __init__(
        self, epochs, batch_size, valid_size, num_inputs, num_outputs, lr, weight_decay
    ) -> None:
        super().__init__()

        # 1 2_conv_2_linear_paper
        # self.backbone = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=3, padding=1), # 32 x 32 x 32
        #     nn.MaxPool2d(2, 2), # 32 x 16 x 16
        #     nn.ReLU(),

        #     nn.Conv2d(32, 32, kernel_size=3, padding=1), # 32 x 16 x 16
        #     nn.MaxPool2d(2, 2), # 32 x 8 x 8
        #     nn.ReLU(),
        # )

        # 2 3_conv_3_linear
        # self.backbone = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=3, padding=1), # 32 x 32 x 32
        #     nn.MaxPool2d(2, 2), # 32 x 16 x 16
        #     nn.ReLU(),

        #     nn.Conv2d(32, 64, kernel_size=3, padding=1), # 64 x 16 x 16
        #     nn.MaxPool2d(2, 2), # 64 x 8 x 8
        #     nn.ReLU(),

        #     nn.Conv2d(64, 128, kernel_size=3, padding=1), # 128 x 8 x 8
        #     nn.MaxPool2d(2, 2), # 128 x 4 x 4
        #     nn.ReLU(),
        # )

        # # 3 4_conv_batch_3_linear
        self.backbone = nn.Sequential(
            nn.Conv2d(num_inputs, 32, kernel_size=3, padding=1),  # 32 x 32 x 32
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),  # 32 x 16 x 16
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64 x 16 x 16
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),  # 64 x 8 x 8
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 128 x 8 x 8
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),  # 128 x 4 x 4
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 256 x 4 x 4
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),  # 256 x 2 x 2
            nn.ReLU(),
        )

        # # 4 5_conv_batch_3_linear
        # self.backbone = nn.Sequential(
        #     nn.Conv2d(num_inputs, 16, kernel_size=3, padding=1), # 32 x 32 x 32
        #     nn.BatchNorm2d(16),
        #     nn.MaxPool2d(2, 2), # 32 x 16 x 16
        #     nn.ReLU(),

        #     nn.Conv2d(16, 32, kernel_size=3, padding=1), # 32 x 32 x 32
        #     nn.BatchNorm2d(32),
        #     nn.MaxPool2d(2, 2), # 32 x 16 x 16
        #     nn.ReLU(),

        #     nn.Conv2d(32, 64, kernel_size=3, padding=1), # 64 x 16 x 16
        #     nn.BatchNorm2d(64),
        #     nn.MaxPool2d(2, 2), # 64 x 8 x 8
        #     nn.ReLU(),

        #     nn.Conv2d(64, 128, kernel_size=3, padding=1), # 128 x 8 x 8
        #     nn.BatchNorm2d(128),
        #     nn.MaxPool2d(2, 2), # 128 x 4 x 4
        #     nn.ReLU(),

        #     nn.Conv2d(128, 256, kernel_size=3, padding=1), # 256 x 4 x 4
        #     nn.BatchNorm2d(256),
        #     nn.MaxPool2d(2, 2), # 256 x 2 x 2
        #     nn.ReLU(),
        # )

        self.flatten = nn.Flatten()

        # 1 2_conv_2_linear_paper
        # self.head = nn.Sequential(
        #     nn.Linear(32 * 8 * 8, 64), # 2048 x 64
        #     nn.ReLU(),

        #     nn.Linear(64, 1), # 64 x 1
        #     nn.Sigmoid()
        # )

        # 2 3_conv_3_linear
        # self.head = nn.Sequential(
        #     nn.Linear(128 * 4 * 4, 1024), # 2048 x 1024
        #     nn.ReLU(),

        #     nn.Linear(1024, 256), # 1024 x 256
        #     nn.ReLU(),

        #     nn.Linear(256, 1), # 256 x 1
        #     nn.Sigmoid()
        # )

        # 3 4_conv_batch_3_linear
        self.head = nn.Sequential(
            nn.Linear(256 * 2 * 2, 512),  # 2048 x 1024
            nn.ReLU(),
            nn.Linear(512, 128),  # 1024 x 128
            nn.ReLU(),
            nn.Linear(128, num_outputs),  # 128 x 1
            nn.Sigmoid(),
        )

        # 4 5_conv_batch_3_linear
        # self.head = nn.Sequential(
        #     nn.Linear(256 * 1 * 1, 128), # 2048 x 1024
        #     nn.ReLU(),

        #     nn.Linear(128, 64), # 1024 x 128
        #     nn.ReLU(),

        #     nn.Linear(64, num_outputs), # 128 x 1
        #     nn.Sigmoid()
        # )

        self.cross_entropy = nn.CrossEntropyLoss()
        self.my_accuracy = MyAccuracy()
        # self.accuracy = torchmetrics.Accuracy(task='binary', num_classes=2)
        self.f1_score = torchmetrics.F1Score(task="binary", num_classes=2)

        self.epochs = epochs
        self.batch_size = batch_size
        self.valid_size = valid_size
        self.lr = lr
        self.weight_decay = weight_decay

        self.train_loss = 0.0
        self.valid_loss = 0.0

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.head(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, y_hat, y = self._common_step(batch, batch_idx)
        # accuracy = self.my_accuracy(y_hat, y)
        # f1_score = self.f1_score(y_hat, y)
        # slow; create an training_end function to calculate at end
        self.log_dict({"train_loss": loss}, on_step=False, on_epoch=True, prog_bar=True)

        if batch_idx % 100 == 0:
            x = x[:8]
            grid = torchvision.utils.make_grid(x.view(-1, 1, 32, 32))
            self.logger.experiment.add_image("CIFAKE_images", grid, self.global_step)

        # self.logger.experiment.add_scalars('Train vs Valid', {'train_loss': loss}, self.global_step)
        self.train_loss += (1 / (batch_idx + 1)) * (loss.data.item() - self.train_loss)

        return {"loss": loss, "y_hat": y_hat, "y": y}

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        y_hats = torch.cat([x["y_hat"] for x in outputs])
        ys = torch.cat([x["y"] for x in outputs])
        self.log_dict(
            {
                "train_accuracy": self.my_accuracy(y_hats, ys),
                "train_f1_score": self.f1_score(y_hats, ys),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.logger.experiment.add_scalars(
            "Train vs Valid", {"train_loss": self.train_loss}, self.global_step
        )
        self.train_loss = 0.0

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx)
        self.log("valid_loss", loss)

        self.valid_loss += (1 / (batch_idx + 1)) * (loss.data.item() - self.valid_loss)
        # self.logger.experiment.add_scalars('Train vs Valid', {'valid_loss': self.valid_loss}, self.global_step)
        return loss

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.logger.experiment.add_scalars(
            "Train vs Valid", {"valid_loss": self.valid_loss}, self.global_step
        )
        self.valid_loss = 0.0

    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx)
        self.log("test_loss", loss)
        return {"loss": loss, "y_hat": y_hat, "y": y}

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        y_hats = torch.cat([x["y_hat"] for x in outputs])
        ys = torch.cat([x["y"] for x in outputs])
        self.log_dict(
            {
                "test_accuracy": self.my_accuracy(y_hats, ys),
                "test_f1_score": self.f1_score(y_hats, ys),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

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
        optimizer1 = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer1,
        )
        return {
            "optimizer": optimizer1,
            "lr_scheduler": scheduler1,
            "monitor": "valid_loss",
        }
