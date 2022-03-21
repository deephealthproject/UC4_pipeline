import argparse
import pandas as pd
import os
import numpy as np
import json
import torch
from matplotlib import cm
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import dataloaders
import models
import pandas as pd
from utils import losses
from utils.metrics import eval_metrics
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.size' : 16
})

def is_nan(x):
    return (x != x)

def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT 
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def white_to_color(img, color=(255, 0, 0)):
    pixels = img.load() # create the pixel map
    for i in range(img.size[0]):    # for every col:
        for j in range(img.size[1]):    # For every row
            if pixels[i,j] == (255, 255, 255):
                pixels[i,j] = color # set the colour accordingly

def colorize_mask(mask, palette):

    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    new_mask = Image.fromarray(mask.astype(np.uint8))
    new_mask.putpalette(palette)
    return new_mask

def create_heatmap(img):
    return Image.fromarray(np.uint8(cm.inferno(img)*255))

def save_images(image, pred_mask, gt_mask, output_path, image_file, palette, mode='heatmap', iou=None, dice_score=None, c=0):
	# Saves the image, the model output and the results after the post processing

    image = image.convert("RGB") 
    image_np = np.array(image, copy=False)
    pred_np = np.array(pred_mask, copy=False)*255
    gt_np = np.array(gt_mask, copy=False)

    #image_np = image_np[..., np.newaxis]
    #print(image_np.shape)
    pred_np = pred_np.squeeze()
    gt_np = gt_np.squeeze()
    # Prediction summed in R channel
    image_np[:, :, 0] = np.where(pred_np == 255, pred_np,
                                    image_np[:, :, -1])
    # Ground truth summed in G channel
    image_np[:, :, 1] = np.where(gt_np == 255, gt_np,
                                    image_np[:, :, 1])

    im = Image.fromarray(image_np)
    # w, h = image.size
    # if w != 512:
    #     print(image_file)
    #     print(image.size)
    # image = image.resize((512,512))
    # w, h = image.size

    # new_im = Image.new('RGB', (w, h))
    # if gt_mask.size[0] != 512:
    #     print(gt_mask.size)
    #     print(pred_mask.size)
    #     gt_mask = gt_mask.resize((512, 512))

    image_file = os.path.basename(image_file).split('.')[0]
    patient = image_file.split("_")[0]
    exam = image_file.split("_")[1]
    if  mode == 'heatmap':
        pred_mask = create_heatmap(pred_mask)
    else:
        pred_mask = colorize_mask(pred_mask, palette)
        white_to_color(pred_mask, (255, 0, 0))
    pred_mask = pred_mask.convert('RGB')
    #print(os.path.join(output_path, 'predictions'))
    #print('{}_pred.png'.format(image_file))
    #print(os.path.join(output_path, 'predictions', '{}_pred.png'.format(image_file)))
    path = os.path.join(output_path, 'predictions', "patient_{}".format(patient), "exam_{}".format(exam))
    if not os.path.exists(path):
        os.makedirs(path)
    pred_mask.save(os.path.join(output_path, 'predictions', "patient_{}".format(patient), "exam_{}".format(exam), '{}_{}_pred.png'.format(image_file, c)))
    gt_mask = gt_mask.convert('RGB')
    path = os.path.join(output_path, 'ground_truth', "patient_{}".format(patient), "exam_{}".format(exam))
    if not os.path.exists(path):
        os.makedirs(path)
    gt_mask.save(os.path.join(output_path, 'ground_truth', "patient_{}".format(patient), "exam_{}".format(exam), '{}_{}_gt.png'.format(image_file, c)))
    # gt_mask = gt_mask.convert('RGB')
    # white_to_color(gt_mask, (0, 255, 0))
    # image = image.convert('RGB')
    # pred_im = Image.blend(image, pred_mask, 0.5)
    # gt_im = Image.blend(image, gt_mask, 0.5)

    # new_im.paste(gt_im, (0,0))
    # new_im.paste(pred_im, (0,0))

    #new_im.save(os.path.join(output_path, 'predictions_groundtruth', '{}_gt_pred_mask_iou_{}_dice_score_{}.png'.format(image_file, np.round(iou, 3), np.round(dice_score, 3))))   
    path = os.path.join(output_path, 'predictions_groundtruth', "patient_{}".format(patient), "exam_{}".format(exam))
    if not os.path.exists(path):
        os.makedirs(path)
    im.save(os.path.join(output_path, 'predictions_groundtruth', "patient_{}".format(patient), "exam_{}".format(exam),'{}_{}_gt_pred_mask_iou_{}_dice_score_{}.png'.format(image_file, c, np.round(iou, 3), np.round(dice_score, 3))))   
    
    # output_im = Image.new('RGB', (w*2, h))
    # output_im.paste(image, (0,0))
    # output_im.paste(colorized_mask, (w,0))
    # output_im.save(os.path.join(output_path, image_file+'_colorized.png'))
    # mask_img = Image.fromarray(mask, 'L')
    # mask_img.save(os.path.join(output_path, image_file+'.png'))

def main():
    args = parse_arguments()
    config = json.load(open(args.config))
    config["test_loader"]["args"]["batch_size"] = 1
    #config["loss"] = "GeneralizedDiceLoss"
    # config['val_loader']['args']['mean'] = 0.2603
    # config['val_loader']['args']['std'] = 0.3001
    #config['val_loader']['args']['split'] = 'test'

    # Dataset used for training the model
    config['test_loader']['args']['data_dir'] = args.data
    config['test_loader']['args']['val'] = True
    loader = get_instance(dataloaders, 'test_loader', config)
    print(len(loader.dataset))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(config['test_loader']['args']['mean'], config['test_loader']['args']['std'])
    num_classes = 1
    palette = loader.dataset.palette

    # Model
    if config["arch"]["type"] == "UNet3D":
        model = get_instance(models, 'arch', config)
    else:
        model = get_instance(models, 'arch', config, config['num_classes'], config['in_channels'])
    availble_gpus = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')

    checkpoint = torch.load(args.model)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
       checkpoint = checkpoint['state_dict']
    if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
       model = torch.nn.DataParallel(model, availble_gpus)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # get function handles of loss and metrics
    loss_fn = getattr(losses, config['loss'])(ignore_index = config['ignore_index'])
    #metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # total_loss = []
    # total_correct, total_label = 0, 0
    # tbar = tqdm(loader, ncols=130)
    # total_iou = []
    # with torch.no_grad():
    #     for i, (data, target) in enumerate(tbar):
    #         data, target = data.to(device), target.to(device)
    #         output = model(data)
    #         #print(output.size())
    #         #print(torch.mean(output[0]))

    #         loss = loss_fn(output, target, threshold=0.50)
    #         #print(loss)
    #         #if isinstance(loss, torch.nn.DataParallel):
    #         #    loss = loss.mean()
    #         total_loss.append(loss)
    #         #print(total_loss)

    #         seg_metrics = eval_metrics(output, target, num_classes)
    #         #print(output.size())
    #         #print(target.size())
    #         total_correct += seg_metrics[0]
    #         total_label += seg_metrics[1]
    #         total_iou.append(seg_metrics[2])
    #         #print(seg_metrics[2])
    #         #total_iou += seg_metrics[2]
    #         #print(total_inter)
    #         #print(total_union)
    #         #if i==1:
    #         #    break
    #     pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
    #     #IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
    #     #print(IoU.size())
    #     #mIoU = torch.cat(total_iou, 0).mean().item()
    #     mIoU = torch.tensor(total_iou).mean().item()
    #     mDiceScore = 1-torch.cat(total_loss, 0).mean().item()
    #     #mIoU = total_iou
    #     metrics = {
    #         "Pixel_Accuracy": np.round(pixAcc, 3),
    #         "Mean_IoU": np.round(mIoU, 3),
    #         "Mean Dice Score": np.round(mDiceScore, 3)
    #     }
    #     print(metrics)
    #     #print("iou "+str(iou_scores.mean))

    if not os.path.exists(args.output):
        os.makedirs(args.output)
        os.makedirs(os.path.join(args.output, 'predictions'))
        os.makedirs(os.path.join(args.output, 'ground_truth'))
        os.makedirs(os.path.join(args.output, 'predictions_groundtruth'))

    gt_masks_files = loader.dataset.masks_paths
    image_files = loader.dataset.images_paths
    total_loss = []
    total_iou = []
    output_df = pd.DataFrame(columns=['image','iou_score','dice_score'])
    y_true = []
    y_pred = []
    target_names = ['Background', 'Nodule']

    if config["arch"]["type"] == "UNet3D":
        with torch.no_grad():
            tbar = tqdm(image_files, ncols=100)
            for i, img_file in enumerate(tbar):
                images_np_chunks = np.load(img_file).astype(np.float32)
                images_np_chunks = np.transpose(images_np_chunks, (1, 2, 0))
                gt_np_chunks = np.load(gt_masks_files[i]).astype(np.int32)/255
                gt_np_chunks = np.transpose(gt_np_chunks, (1, 2, 0))

                num_chunks = images_np_chunks.shape[2]
                
                image = images_np_chunks
                gt_mask = gt_np_chunks
                input = normalize(to_tensor(image)).unsqueeze(0)
                
                logits = model(input.to(device))
                loss = loss_fn(logits, to_tensor(gt_mask).to(device), threshold=0.5)
                total_loss.append(loss)
                seg_metrics = eval_metrics(logits, to_tensor(gt_mask).to(device), num_classes)
                current_iou = seg_metrics[2].mean().item()
                dice_score = 1-loss.mean().item()
                total_iou.append(seg_metrics[2])
                #print(current_iou)
                #print(total_iou.mean)
                #iou = seg_metrics[2] / (np.spacing(1) + seg_metrics[3])
                logits = logits.squeeze(1).cpu().numpy()
                prob = torch.sigmoid(torch.from_numpy(logits))
                if args.mode =='heatmap':
                    prediction = prob.squeeze(0).cpu().numpy()
                else:
                    prediction = (prob > 0.5).type(torch.long).squeeze(0).cpu().numpy()
                
                for c in range(num_chunks):
                    output_df = output_df.append({'image': img_file, 'iou_score': current_iou, 'dice_score': dice_score}, ignore_index=True)
                    #print(image[:,:,c].shape)
                    #print(gt_mask[:,:,c].shape)
                    #print(prediction[:,:,c].shape)
                    im_c =  Image.fromarray(image[:,:,c])
                    gt_c = Image.fromarray(gt_mask[:,:,c]*255)
                    save_images(im_c, prediction[c,:,:], gt_c, args.output, img_file, palette, args.mode, current_iou, dice_score, c)
    else:
        with torch.no_grad():
            tbar = tqdm(image_files, ncols=100)
            for i, img_file in enumerate(tbar):
                image = Image.open(img_file)
                image_shape = np.asarray(image).shape
                if is_nan(gt_masks_files[i]):
                    gt_mask = np.zeros(image_shape, dtype=np.int32)
                else:
                    gt_mask = np.asarray(Image.open(gt_masks_files[i]), dtype=np.int32)/255
                input = normalize(to_tensor(image)).unsqueeze(0).to(device)
                
                logits = model(input.to(device))

                predictions = torch.sigmoid(logits)
                predictions = (predictions > 0.5).cpu().numpy()
                y_true.append(np.max(gt_mask))
                y_pred.append(np.max(predictions))

                loss = loss_fn(logits, to_tensor(gt_mask).to(device), threshold=0.5)
                total_loss.append(loss)
                seg_metrics = eval_metrics(logits, to_tensor(gt_mask).to(device), num_classes)
                current_iou = seg_metrics[2].mean().item()
                dice_score = 1-loss.mean().item()
                total_iou.append(seg_metrics[2])
                #print(current_iou)
                #print(total_iou.mean)
                #iou = seg_metrics[2] / (np.spacing(1) + seg_metrics[3])
                logits = logits.squeeze(1).cpu().numpy()
                prob = torch.sigmoid(torch.from_numpy(logits))
                if args.mode =='heatmap':
                    prediction = prob.squeeze(0).cpu().numpy()
                else:
                    prediction = (prob > 0.5).type(torch.long).squeeze(0).cpu().numpy()
                output_df = output_df.append({'image': img_file, 'iou_score': current_iou, 'dice_score': dice_score}, ignore_index=True)
                #if i==100:
                #    break
                #print(img_file)
                save_images(image, prediction, Image.fromarray(gt_mask*255), args.output, img_file, palette, args.mode, current_iou, dice_score)
    #IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
    #print(IoU.size())
    mIoU = torch.cat(total_iou, 0).mean().item()
    mDiceScore = 1-torch.cat(total_loss, 0).mean().item()
    #mIoU = total_iou
    metrics = {
        "IoU": np.round(mIoU, 3),
        "Dice score": np.round(mDiceScore,3)
    }
    print(metrics)
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    print("TN: {}".format(cm[0,0]))
    print("FN: {}".format(cm[1,0]))
    print("TP: {}".format(cm[1,1]))
    print("FP: {}".format(cm[0,1]))
    print(classification_report(y_true, y_pred, target_names=target_names))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap="Blues")
    #disp.ax_.set_title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, "confusion_matrix_normalized.pdf"))
    output_df.to_csv(os.path.join(args.output, 'results.csv'))

def parse_arguments():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--config', default='/data/deephealth/deephealth-uc4/data/saved/UNet2D/pretrained_blackmasks/01-18_13-06/config.json',type=str,
                        help='The config used to train the model')
    parser.add_argument('-m', '--model', default='/data/deephealth/deephealth-uc4/data/saved/UNet2D/pretrained_blackmasks/01-18_13-06/best_model10.pth', type=str,
                        help='Path to the .pth model checkpoint to be used in the prediction')
    parser.add_argument('-i', '--data', default='/data/deephealth/deephealth-uc4/data/processed/unitochest/test', type=str,
                        help='Path to the images to be segmented')
    parser.add_argument('-o', '--output', default='/data/deephealth/deephealth-uc4/data/saved/UNet2D/pretrained_blackmasks/01-18_13-06/unitochest/test_new', type=str,  
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
  
#python test.py --config saved/UNet/06-01_14-51/config.json --model saved/UNet/06-01_14-51/best_model.pth --data data/processed/dataset_molinette20210418_np_lungs_mediastinal/test/ --output data/outputs/UNet/06-01_14-51/test
#python test.py --config saved/UNet_dataset_molinette20210418_correct/06-16_18-33/config.json --model saved/UNet_dataset_molinette20210418_correct/06-16_18-33/model.pth --data /mnt/16c983f0-1f80-4125-846d-2a2f5969cdad/deephealth-uc4/data/processed/dataset_molinette20210418_correct/test --output data/outputs/UNet_dataset_molinette20210418_correct/06-16_18-33/test

if __name__ == '__main__':
    main()

