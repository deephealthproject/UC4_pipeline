import argparse
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

def white_to_color(img, color=(255, 0, 0)):
    pixels = img.load() # create the pixel map
    for i in range(img.size[0]):    # for every col:
        for j in range(img.size[1]):    # For every row
            if pixels[i,j] == (255, 255, 255):
                pixels[i,j] = color # set the colour accordingly

def create_heatmap(img):
    return Image.fromarray(np.uint8(cm.inferno(img)*255))

def colorize_mask(mask, palette):

    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    new_mask = Image.fromarray(mask.astype(np.uint8))
    new_mask.putpalette(palette)
    return new_mask

def save_images(image, pred_mask, output_path, image_file, palette, mode='heatmap'):
	# Saves the image, the model output and the results after the post processing
    w, h = image.size

    new_im = Image.new('RGB', (w, h))

    image_file = os.path.basename(image_file).split('.')[0]

    if  mode == 'heatmap':
        pred_mask = create_heatmap(pred_mask)
    else:
        pred_mask = colorize_mask(pred_mask, palette)
        white_to_color(pred_mask, (255, 0, 0))

    pred_mask = pred_mask.convert('RGB')
    image = image.convert('RGB')
    pred_im = Image.blend(image, pred_mask, 0.5)

    new_im.paste(pred_im, (0,0))

    new_im.save(os.path.join(output_path, image_file+'_gt_pred_mask.png'))
    # output_im = Image.new('RGB', (w*2, h))
    # output_im.paste(image, (0,0))
    # output_im.paste(colorized_mask, (w,0))
    # output_im.save(os.path.join(output_path, image_file+'_colorized.png'))
    # mask_img = Image.fromarray(mask, 'L')
    # mask_img.save(os.path.join(output_path, image_file+'.png'))

def main():
    args = parse_arguments()
    config = json.load(open(args.config))

    # Get mean and std used for training the model
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(config['train_loader']['args']['mean'], config['train_loader']['args']['std'])
    num_classes = 1
    palette = [0,0,0,255,255,255]

    # Model
    model = getattr(models, config['arch']['type'])(num_classes, **config['arch']['args'])
    availble_gpus = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')

    checkpoint = torch.load(args.model)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    image_dir = args.data
    image_files = sorted([path for path in Path(image_dir).rglob('*.png')])

    with torch.no_grad():
        tbar = tqdm(image_files, ncols=100)
        for i, img_file in enumerate(tbar):
            image = Image.open(img_file)
            input = normalize(to_tensor(image)).unsqueeze(0)
            
            logits = model(input.to(device))
            logits = logits.squeeze(1).cpu().numpy()
            prob = torch.sigmoid(torch.from_numpy(logits))
            if args.mode =='heatmap':
                prediction = prob.squeeze(0).cpu().numpy()
            else:
                prediction = (prob >= 0.5).type(torch.long).squeeze(0).cpu().numpy()
            save_images(image, prediction, args.output, img_file, palette, args.mode)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--config', default='saved/UNet/10-14_00-07/config.json',type=str,
                        help='The config used to train the model')
    parser.add_argument('-m', '--model', default='saved/UNet/10-14_00-07/best_model.pth', type=str,
                        help='Path to the .pth model checkpoint to be used in the prediction')
    parser.add_argument('-i', '--data', default='data/processed/patient_002/', type=str,
                        help='Path to the images to be segmented')
    parser.add_argument('-o', '--output', default='data/outputs/UNet/10-14_00-07_p2/test', type=str,  
                        help='Output Path')
    parser.add_argument('-e', '--extension', default='png', type=str,
                        help='The extension of the images to be segmented')
    parser.add_argument('-mode', '--mode', default='binary', type=str,
                        help='Binary or heatmap output mask image')
    args = parser.parse_args()
    return args

#python test.py --config saved/UNet/06-24_01-27/config.json --model saved/UNet/06-24_01-27/best_model.pth --data data/processed/dataset_molinette20200619/test/ --output data/outputs/UNet/06-24_01-27/test
#python test.py --config saved/UNet/07-16_23-24/config.json --model saved/UNet/07-16_23-24/best_model.pth --data data/processed/dataset_molinette20200716/test/ --output data/outputs/UNet/07-16_23-24/test
#python test.py --config saved/UNet/07-21_14-08/config.json --model saved/UNet/07-21_14-08/best_model.pth --data data/processed/dataset_molinette_trainvaltest/test/ --output data/outputs/UNet/07-21_14-08/test

if __name__ == '__main__':
    main()

