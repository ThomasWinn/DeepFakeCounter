import torch
import torch.nn as nn

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