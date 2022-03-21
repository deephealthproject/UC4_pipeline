
import numpy as np
import os
from base import BaseDataLoader, BaseDataSet3D
from utils import palette
from pathlib import Path

class MolinetteLungsDataset3D(BaseDataSet3D):
    def __init__(self, **kwargs):
        self.num_classes = 1
        self.palette = palette.MolinetteLungs_palette
        super(MolinetteLungsDataset3D, self).__init__(**kwargs)

    def _set_files(self):
        self.image_dir = os.path.join(self.root, 'images')#, self.split)
        self.label_dir = os.path.join(self.root, 'ground_truth')#, self.split)
        self.images_paths = sorted([path for path in Path(self.image_dir).rglob('*.npy')])
        self.masks_paths = sorted([path for path in Path(self.label_dir).rglob('*.npy')])
        assert(len(self.masks_paths) == len(self.images_paths))

    
    def _load_data(self, index):
        image_path = self.images_paths[index]
        label_path = self.masks_paths[index]
        #image = np.asarray(Image.open(image_path), dtype=np.float32)
        image = np.load(image_path).astype(np.float32)
        image = np.transpose(image, (1, 2, 0))
        #image = np.expand_dims(image, axis=0)
        #print(image.shape)
        label = np.load(label_path).astype(np.int32)/255
        label = np.transpose(label, (1, 2, 0))
        #print("ciao")
        #print(label.shape)
        #label = np.expand_dims(label, axis=0)
        #print(label.shape)
        #label = np.asarray(Image.open(label_path), dtype=np.int32)/255
        image_id = index
        #label -= 1
        #label[label == -1] = 255
        #label -= 1
        return image, label, image_id
    
    def __len__(self):
        return len(self.images_paths)

class MolinetteLungsLoader3D(BaseDataLoader):
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
        self.dataset = MolinetteLungsDataset3D(**kwargs)
        super(MolinetteLungsLoader3D, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split=val_split)