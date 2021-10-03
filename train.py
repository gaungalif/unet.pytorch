import argparse
import pytorch_lightning as pl
from unet.datamodule import BrainMRISegmentationDataModule
from unet.module import UNet
from pytorch_lightning.callbacks import ModelCheckpoint
# import mlflow
from pathlib import Path
import torch
import logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    
    
    parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    
    dict_args = vars(hparams)
    
    # mlflow.set_tracking_uri("http://localhost:54849")
    # mlflow.pytorch.autolog()
    check_dir = hparams.checkpoint_dir
    check_dir = Path(check_dir)
    datamod = BrainMRISegmentationDataModule(**dict_args)
    mobilenetv2 = UNet(**dict_args)
    
    model_checkpoint = ModelCheckpoint(
        dirpath=check_dir,
        save_top_k=1,
        filename="mobilenet_v2-{epoch:02d}-{val_step_loss:.4f}-{val_step_acc:.4f}",
        verbose=True,
        monitor='val_step_loss',
        mode='min',
    )
    trainer = pl.Trainer.from_argparse_args(hparams, callbacks=model_checkpoint)
    # with mlflow.start_run() as run:
    trainer.fit(mobilenetv2, datamod)
    trainer.save_checkpoint(check_dir.joinpath('latest.ckpt'))
    # mobilenetv2.model.state_dict()
    
    metrics =  trainer.logged_metrics
    # metrics['epoch']
    vacc, vloss, last_epoch = metrics['val_step_acc'], metrics['val_step_loss'], metrics['epoch']
    
    filename = f'mobilenet_v2-{last_epoch:02d}_acc{vacc:.4f}_loss{vloss:.4f}.pth'
    saved_filename = str(Path('classify.pytorch/weights').joinpath(filename))
    
    logging.info(f"Prepare to save training results to path {saved_filename}")
    torch.save(mobilenetv2.model.state_dict(), saved_filename)