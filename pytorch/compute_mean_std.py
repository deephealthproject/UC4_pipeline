import argparse
import pandas as pd
import scipy
import os
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from matplotlib import cm
from torchvision import transforms
from scipy import ndimage
from tqdm import tqdm
from math import ceil
from glob import glob
from PIL import Image
import dataloaders
import models
from utils import losses
from utils.metrics import eval_metrics, AverageMeter
from dataloaders.molinette import MolinetteLungsLoader
from dataloaders.molinette3D import MolinetteLungsLoader3D

def main(args):  
    loader = MolinetteLungsLoader(data_dir=os.path.join(args.data, 'train'),
                            batch_size=128, 
                            num_workers=0, 
                            augment=True, 
                            base_size=512, 
                            scale=False,
                            shuffle=False, 
                            split="train")  
    # loader = MolinetteLungsLoader3D(data_dir=os.path.join(args.data, 'train'),
    #                         batch_size=16, 
    #                         num_workers=4, 
    #                         augment=True, 
    #                         base_size=512, 
    #                         scale=False,
    #                         shuffle=False, 
    #                         split="train")
    mean = 0.
    std = 0.
    var = 0.
    nb_samples = 0.
    print(len(loader))
    background_pixels = 0
    foreground_pixels = 0
    for data, targets in tqdm(loader):
        #print(data.size())
        #print(data.min())
        #print(data.max())
        batch_samples = data.size(0)*data.size(1)
        h, w = data.size(-1), data.size(-2)
        #print(len(data.size()))
        #channels = data.size(1)
        #data = data.view(batch_samples,channels, -1)
        data = data.view(batch_samples, 1, h*w)
        background_pixels += (targets == 0.).sum()
        foreground_pixels += (targets == 1.).sum()
        mean += data.mean(2).sum(0)
        var += data.var(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    var /= nb_samples
    std = torch.sqrt(var)
    print("mean: " + str(mean))
    print("std: " + str(std))
    print("background pixels: " + str(background_pixels))
    print("foreground pixels: " + str(foreground_pixels))
    print("pos_weight: " + str(background_pixels/foreground_pixels))
    print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute mean and std of a given dataset')
    parser.add_argument('-c', '--config', default='saved/UNet_dataset_molinette20210418_correct/12-01_23-44/config.json',type=str,
                        help='The config used to train the model')
    parser.add_argument('-i', '--data', default='/data/deephealth-uc4/data/processed/unitochest/', type=str,
                        help='Path to the images to be segmented')
    args = parser.parse_args()
    main(args)
