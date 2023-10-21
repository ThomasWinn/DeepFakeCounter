import lightning.pytorch as pl
from torch.utils.tensorboard import SummaryWriter

from data.dataloader import Data_DataLoader
from models.cnn import pl_CIFAKE_CNN

batch_size = 32

loader_obj = Data_DataLoader('data/mean_and_std.pt', batch_size=batch_size, valid_size=0.2, num_workers=0)
train_loader, valid_loader = loader_obj.get_train_val_dataloader()
test_loader = loader_obj.get_test_dataloader()

model = pl_CIFAKE_CNN()

# writer = SummaryWriter() # default directory =  runs/**CURRENT_DATETIME_HOSTNAME**

# trainer = pl.Trainer(limit_train_batches=100, max_epochs=1)
# trainer.fit(model=model, train_dataloaders=train_loader)

# trainer = pl.Trainer()
# trainer.test(model=model, dataloaders=test_loader)

# train with both splits
trainer = pl.Trainer()
trainer.fit(model, train_loader, valid_loader)