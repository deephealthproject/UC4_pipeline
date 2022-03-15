
import numpy as np
import os
from scipy import ndimage
import torch
import cv2
from tqdm import tqdm
from PIL import Image
from glob import glob
from base import BaseDataSet, BaseDataLoader, BaseDataSetClassification
from utils import palette
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms

def is_nan(x):
    return (x != x)

class MolinetteLungsDataset(BaseDataSetClassification):
    def __init__(self, **kwargs):
        self.num_classes = 1
        self.palette = palette.MolinetteLungs_palette
        super(MolinetteLungsDataset, self).__init__(**kwargs)

    def _set_files(self):
        
        self.image_dir = os.path.join(self.root, 'images')#, self.split)
        self.label_dir = os.path.join(self.root, 'masks')#, self.split)
        self.dataframe['image'] = self.dataframe['image'].apply(lambda x: os.path.join(self.image_dir, x))
        self.dataframe['mask'] = self.dataframe['mask'].apply(lambda x: x if is_nan(x) else os.path.join(self.label_dir, x))
        self.images_paths = list(self.dataframe['image'])
        self.masks_paths = list(self.dataframe['mask'])
        assert(len(self.masks_paths) == len(self.images_paths))
    
    def _load_data(self, index):
        image_path = self.images_paths[index]
        label_path = self.masks_paths[index]
        image = np.asarray(Image.open(image_path), dtype=np.float32)
        if is_nan(label_path):
            label = 0
        else:
            label = 1
        image_id = index
        return image, label, image_id
    
    def __len__(self):
        return len(self.images_paths)

class MolinetteLungsClassificationLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, mean=[0], std=[1], crop_size=None, base_size=None, scale=True, num_workers=1, val=False,
                    shuffle=False, flip=False, rotate=False, blur= False, augment=False, val_split= None, return_id=False):

        self.MEAN = mean
        self.STD = std

        print((self.MEAN, self.STD))

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }
        self.dataset = MolinetteLungsDataset(**kwargs)
        super(MolinetteLungsClassificationLoader, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split=val_split)