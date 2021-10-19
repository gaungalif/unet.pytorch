import pandas as pd
import torch
from pathlib import Path

from torch.utils.data import DataLoader

from PIL import Image
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from torchvision import transforms
from torchvision.datasets import ImageFolder

from typing import *

import unet.transforms as T

def transform_fn(train=False, size=(224,224)):
    normalize = T.PairNormalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225]) 

    
    if train:                                 
        return T.PairCompose([
            T.PairResize(size),
            T.PairRandomRotation(20),
            T.PairToTensor(),
            # normalize,
        ])
    else:
        return T.PairCompose([
            T.PairResize(size),
            T.PairToTensor(),
            # normalize,
        ])

class BrainMRISegmentationDataset(ImageFolder):
    def __init__(self, root: Path, train: bool = True, val_size: float = 0.2, transform=None, 
                 image_transform=None, mask_transform=None, **kwargs):
        super(BrainMRISegmentationDataset, self).__init__(Path(root), transform)
        self.train = train
        self.val_size = val_size
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        
        
        self.dataframe = self._load_dataframe()
        self._split_train_valid()
        
        
    def _load_dataframe(self):
        files = sorted(list(self.root.glob("*/*mask.tif")))
        data = { 'name': [], 'image_path': [], 'mask_path': []}
        for idx in range(len(files)):
            parent_dir = files[idx].parent.name
            basename = files[idx].name.split("_mask.tif")[0]
            img_name = f'{basename}.tif'
            msk_name = f'{basename}_mask.tif'

            data['name'].append(parent_dir)
            data['image_path'].append(f'{parent_dir}/{img_name}')
            data['mask_path'].append(f'{parent_dir}/{msk_name}')

        return pd.DataFrame(data)
    
        
    def _split_train_valid(self):
        train_df, valid_df = train_test_split(self.dataframe, test_size=self.val_size, random_state=1261)
        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)
        
        if self.train:
            self.dataframe = train_df
        else:
            self.dataframe = valid_df
    
    def _load_image(self, path: Path, to_gray=False):
        if not to_gray:
            image = Image.open(path).convert('RGB')
        else:
            image = Image.open(path).convert('L')


            
        return image
        
    def _load_image_mask_path(self, idx):
        data = self.dataframe.iloc[idx]
        img_path = self.root.joinpath(data['image_path'])
        msk_path = self.root.joinpath(data['mask_path'])
        return img_path, msk_path
    
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path, msk_path = self._load_image_mask_path(idx)
        image = self._load_image(img_path)
        mask = self._load_image(msk_path, to_gray=True)
        
        if self.transform:
            img, msk = self.transform(image, mask)

        return img, msk
        
class BrainMRISegmentationDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 1, num_workers: int = 2, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = transform_fn(train=True) 
        self.valid_transform = transform_fn(train=False) 

        self._setup()

    def _setup(self, stage: Optional[str] = None):
            self.brain_trainset = BrainMRISegmentationDataset(root=self.data_dir, train=True, transform=self.train_transform)
            self.brain_validset = BrainMRISegmentationDataset(root=self.data_dir, train=False, transform=self.valid_transform)
   
    def train_dataloader(self):
        return DataLoader(self.brain_trainset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.brain_validset, batch_size=self.batch_size, num_workers=self.num_workers)
   