from torch.utils.tensorboard import SummaryWriter

from data.dataloader import Data_DataLoader

batch_size = 32

loader_obj = Data_DataLoader(batch_size=batch_size, valid_size=0.2, num_workers=0)
train_loader, valid_loader = loader_obj.get_train_val_dataloader()
test_loader = loader_obj.get_test_dataloader()

writer = SummaryWriter() # default directory =  runs/**CURRENT_DATETIME_HOSTNAME**

