import torch
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
