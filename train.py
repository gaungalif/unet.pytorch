from pathlib import Path
import logging
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from unet.datamodule import BrainMRISegmentationDataModule
from unet.module import UNet
import mlflow
from pytorch_lightning.callbacks import QuantizationAwareTraining



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./dataset')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--start_feat', type=int, default=32)


    
    
    parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    
    dict_args = vars(hparams)
    
    # mlflow.set_tracking_uri("http://localhost:5000")
    # mlflow.pytorch.autolog()
    
    datamod = BrainMRISegmentationDataModule(**dict_args)
    unet = UNet(**dict_args)
    model_checkpoint = ModelCheckpoint(
        dirpath='checkpoints/',
        save_top_k=1,
        filename="UNet-{epoch:02d}-{val_step_loss:.4f}",
        verbose=True,
        monitor='val_step_loss',
        mode='min',
    )

    
    trainer = pl.Trainer.from_argparse_args(hparams, callbacks=[QuantizationAwareTraining(observer_type='histogram', input_compatible=True), model_checkpoint])
    # with mlflow.start_run() as run:
    trainer.fit(unet, datamod)
    trainer.save_checkpoint("checkpoints/latest.ckpt")
    
    metrics =  trainer.logged_metrics
    vloss = metrics['val_step_loss']
    
    filename = f'unet-loss{vloss:.4f}.pth'
    saved_filename = str(Path('weights').joinpath(filename))
    
    logging.info(f"Prepare to save training results to path {saved_filename}")
    torch.save(unet.model.state_dict(), saved_filename)
    
    
    
    
