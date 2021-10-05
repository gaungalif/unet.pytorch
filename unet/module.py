import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as FM

import torch.optim as optim

import torchmetrics 

from unet.base import *
class UNetEncoder(nn.Module):
    def __init__(self, in_chan, start_feat=64):
        super(UNetEncoder, self).__init__()
        self.out_chan = start_feat * 8
        self.inconv = InConv(in_chan, start_feat)
        self.down1 = DownConv(start_feat, start_feat*2)
        self.down2 = DownConv(start_feat*2, start_feat*4)
        self.down3 = DownConv(start_feat*4, start_feat*8)
        self.down4 = DownConv(start_feat*8, start_feat*8)

    def forward(self, x):
        inc = self.inconv(x)
        dc1 = self.down1(inc)
        dc2 = self.down2(dc1)
        dc3 = self.down3(dc2)
        dc4 = self.down4(dc3)
        return dc4, dc3, dc2, dc1, inc


class UNetDecoder(nn.Module):
    def __init__(self, in_chan, n_classes):
        super(UNetDecoder, self).__init__()
        self.up1 = UpConv(in_chan, in_chan//4)
        self.up2 = UpConv(in_chan//2, in_chan//8)
        self.up3 = UpConv(in_chan//4, in_chan//16)
        self.up4 = UpConv(in_chan//8, in_chan//16)
        self.outconv = OutConv(in_chan//16, n_classes)

    def forward(self, dc4, dc3, dc2, dc1, inc):
        up1 = self.up1(dc4, dc3)
        up2 = self.up2(up1, dc2)
        up3 = self.up3(up2, dc1)
        up4 = self.up4(up3, inc)
        out = self.outconv(up4)
        return out


class UNet(pl.LightningModule):
    def __init__(self, in_chan=3, n_classes=1, start_feat=64, **kwargs):
        super(UNet, self).__init__()
        self.encoder_in_chan = in_chan
        self.decoder_in_chan = start_feat * 16
        self.start_feat = start_feat
        
        self.criterion = nn.BCEWithLogitsLoss()
        
        self.encoder = UNetEncoder(in_chan=self.encoder_in_chan, start_feat=start_feat)
        self.decoder = UNetDecoder(in_chan=self.decoder_in_chan, n_classes=n_classes)
        
        self.trn_loss: torchmetrics.AverageMeter = torchmetrics.AverageMeter()
        self.val_loss: torchmetrics.AverageMeter = torchmetrics.AverageMeter()
            
    def forward_step(self, x):
        dc4, dc3, dc2, dc1, inc = self.encoder(x)
        out = self.decoder(dc4, dc3, dc2, dc1, inc)
        return out
    
    def forward(self, imgs):
        output = self.forward_step(imgs)
        return output

    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()
        
    def shared_step(self, batch, batch_idx):
        images, labels = batch
        
        preds = self.forward(images)
        loss = self.criterion(preds, labels)
        
        return loss
        
    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        trn_loss = self.trn_loss(loss)
        self.log('trn_step_loss', trn_loss, prog_bar=True, logger=True)
        return loss
    
    def training_epoch_end(self, outs):
        self.log('trn_epoch_loss', self.trn_loss.compute(), logger=True)
    
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        val_loss = self.val_loss(loss)
        self.log('val_step_loss', val_loss, prog_bar=True, logger=True)
        return loss
    
    def validation_epoch_end(self, outs):
        self.log('val_epoch_loss', self.val_loss.compute(), logger=True)
        
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.00005)


if __name__ == '__main__':
    model = UNet(in_chan=3, n_classes=1, start_feat=224)
    input = torch.rand(1, 3, 224, 224)
    out = model(input)
    print(out.shape)
