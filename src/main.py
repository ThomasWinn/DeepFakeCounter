import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.tensorboard import SummaryWriter

from data.cifake_data_module import CIFAKEDataModule
from models.cnn import CIFAKE_CNN


if __name__ == "__main__":
    

    batch_size = 32

    # loader_obj = Data_DataLoader('data/mean_and_std.pt', batch_size=batch_size, valid_size=0.2, num_workers=0)
    # train_loader, valid_loader = loader_obj.get_train_val_dataloader()
    # test_loader = loader_obj.get_test_dataloader()
    dm = CIFAKEDataModule(
        cache_file='data/mean_and_std.pt',
        data_dir='../dataset',
        batch_size=32,
        num_workers=0,
        valid_size=0.2
    )

    model = CIFAKE_CNN()

    # logger = TensorBoardLogger(
    #     save_dir='../tensorboard_logs',
    #     name='paper_cnn',
    #     log_graph=True
    # )

    # train with both splits
    trainer = pl.Trainer(
        accelerator='gpu', # turn to cpu on macbook
        default_root_dir='../',
        devices=1, # how many gpus
        enable_checkpointing=True,
        fast_dev_run=False, # A flag to touch everyline of model to uncover bugs easier. Use in develop. On default = False. Runs one train and valid batch and program ends
        gradient_clip_val=None, # default None, but clips to specific val if you see gradiants exploding
        # logger=logger,
        min_epochs=1,
        max_epochs=2,
        sync_batchnorm=False, # Recommended for smaller batch sizes
    ) # use overfit_batches because a good thing to check is if you can overfit batches, if yhou can't somethings wrong
    # auto_lr_find = find best learning rate
    
    # trainer.tune() to find best parameters
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)