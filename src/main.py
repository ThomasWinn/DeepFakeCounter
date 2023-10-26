import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.profilers import PyTorchProfiler
import torch
from torch.utils.tensorboard import SummaryWriter

import config
from data.dataloader import Data_DataLoader
from data.cifake_data_module import CIFAKEDataModule
from models.cnn import CIFAKE_CNN


if __name__ == "__main__":
    logger = TensorBoardLogger(
        save_dir='../tensorboard_logs',
        name='4_conv_batch_3_linear',
        log_graph=True
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor='valid_loss',
        dirpath='../models/',
        filename='{}-{epoch:02d}-{val_loss:.2f}'.format(logger.name),
        save_top_k=3,
        mode='min'
    )
    
    dm = CIFAKEDataModule(
        cache_file=config.CACHE_FILE,
        data_dir=config.DATASET_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        valid_size=config.VALID_SIZE
    )

    model = CIFAKE_CNN(
        epochs=config.MAX_EPOCHS,
        batch_size=config.BATCH_SIZE,
        valid_size=config.VALID_SIZE,
        num_inputs=config.NUM_INPUTS,
        num_outputs=config.NUM_OUTPUTS,
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # train with both splits
    trainer = pl.Trainer(
        accelerator=config.ACCELERATOR,
        default_root_dir=config.LOG_DIR,
        devices=config.DEVICES, # how many gpus
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
        fast_dev_run=False, # A flag to touch everyline of model to uncover bugs easier. Use in develop. On default = False. Runs one train and valid batch and program ends
        gradient_clip_val=None, # default None, but clips to specific val if you see gradiants exploding
        logger=logger,
        min_epochs=config.MIN_EPOCHS,
        max_epochs=config.MAX_EPOCHS,
        profiler='pytorch',
        sync_batchnorm=False, # Recommended for smaller batch sizes
    ) # use overfit_batches because a good thing to check is if you can overfit batches, if yhou can't somethings wrong
    # auto_lr_find = find best learning rate
    
    # trainer.tune() to find best parameters
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)