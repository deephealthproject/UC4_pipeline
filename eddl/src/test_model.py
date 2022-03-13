# Copyright (c) 2020, Università degli Studi di Modena e Reggio Emilia
# (UNIMORE), AImageLab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""\
Pneumothorax segmentation inference example.

More information and checkpoints available at
https://github.com/deephealthproject/use_case_pipeline
"""

import argparse
import numpy as np
import os
import time
import wandb
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
import pyecvl.ecvl as ecvl

import models.utils as utils
from models.models import UNet
from models.models import Nabla

api = wandb.Api()

MEM_CHOICES = ("low_mem", "mid_mem", "full_mem")

def main(args):
    run = api.run(args.wb_run_path)
    num_workers = args.num_workers
    queue_ratio_size = args.queue_ratio_size
    num_classes = 1
    size = [args.shape, args.shape]  # size of images
    if args.shape == 512:
        mean = 0.3266
        std = 0.3551
    else:
        mean = 0
        std = 1
    thresh = 0.5

    if args.runs_dir:
        os.makedirs(os.path.join(args.runs_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(args.runs_dir, "logs"), exist_ok=True)

    in_ = eddl.Input([1, size[0], size[1]])
    out = UNet(in_, num_classes)
    out_sigm = eddl.Sigmoid(out)
    net = eddl.Model([in_], [out_sigm])
    eddl.build(
        net,
        eddl.adam(0.001),
        ["dice"],
        ["dice"],
        eddl.CS_GPU(args.gpu, mem=args.mem) if args.gpu else eddl.CS_CPU(mem=args.mem),
        False
    )
    eddl.summary(net)

    #if not os.path.exists(args.ckpts):
    #    raise RuntimeError('Checkpoint "{}" not found'.format(args.ckpts))
    best_model = wandb.restore(args.ckpts, run_path=args.wb_run_path)
    print(best_model.name)
    eddl.load(net, best_model.name, "bin")

    test_augs = ecvl.SequentialAugmentationContainer([
        ecvl.AugResizeDim(size),
        ecvl.AugToFloat32(255, divisor_gt=255),
        ecvl.AugNormalize(mean, std),
    ])
    dataset_augs = ecvl.DatasetAugmentations([None, None, test_augs])

    print("Reading dataset")
    d = ecvl.DLDataset(args.dataset, 
                       args.batch_size, 
                       dataset_augs, 
                       ctype=ecvl.ColorType.GRAY, 
                       ctype_gt=ecvl.ColorType.GRAY, 
                       num_workers=num_workers,
                       queue_ratio_size=queue_ratio_size)
    x = Tensor([args.batch_size, d.n_channels_, size[0], size[1]])
    y = Tensor([args.batch_size, d.n_channels_gt_, size[0], size[1]])
    
    print("Testing started!")
    d.SetSplit(ecvl.SplitType.test)
    d.ResetBatch(d.current_split_, False)
    num_samples_test = len(d.GetSplit())
    num_batches_test = num_samples_test // args.batch_size

    iou_evaluator = utils.Evaluator()
    iou_evaluator.ResetEval()
    dice_evaluator = utils.Evaluator()
    dice_evaluator.ResetEval()
    start_time = time.time()
    # Spawn the threads
    d.Start()
    for b in range(num_batches_test):
        print("Batch {:d}/{:d} ".format(
            b + 1, num_batches_test), end="", flush=True)
        samples, x, y = d.GetBatch()
        eddl.forward(net, [x])
        output = eddl.getOutput(out_sigm)
        iou = iou_evaluator.BinaryIoU(np.array(output), np.array(y), thresh=thresh)
        dice = dice_evaluator.DiceCoefficient(np.array(output), np.array(y), thresh=thresh)
        print("- Batch IoU: %.6g " % iou, end="", flush=True)
        print("- Batch dice: %.6g " % dice, end="", flush=True)
        print()
        # for k in range(args.batch_size):
        #     pred = output.select([str(k)])
        #     gt = y.select([str(k)])
        #     pred_np, gt_np = np.array(pred, copy=False), np.array(gt, copy=False)
        #     iou = iou_evaluator.BinaryIoU(pred_np, gt_np, thresh=thresh)
        #     dice = dice_evaluator.DiceCoefficient(pred_np, gt_np, thresh=thresh)
        #     print("- Batch IoU: %.6g " % iou, end="", flush=True)
        #     print("- Batch dice: %.6g " % dice, end="", flush=True)

        #     if args.runs_dir:
        #         # Save original image fused together with prediction and
        #         # ground truth
        #         pred_np[pred_np >= thresh] = 1
        #         pred_np[pred_np < thresh] = 0
        #         pred_np *= 255
        #         pred_ecvl = ecvl.TensorToView(pred)
        #         pred_ecvl.colortype_ = ecvl.ColorType.GRAY
        #         pred_ecvl.channels_ = "xyc"
        #         ecvl.ResizeDim(pred_ecvl, pred_ecvl, (512,512),
        #                            ecvl.InterpolationType.nearest)

        #         filename_gt = d.samples_[d.GetSplit()[b*args.batch_size + k]].label_path_
        #         gt_ecvl = ecvl.ImRead(filename_gt,
        #                                 ecvl.ImReadMode.GRAYSCALE)
        #         ecvl.ResizeDim(gt_ecvl, gt_ecvl, (512,512),
        #                            ecvl.InterpolationType.nearest)

        #         filename = d.samples_[d.GetSplit()[b*args.batch_size + k]].location_[0]

        #         # Image as BGR
        #         img_ecvl = ecvl.ImRead(filename)
        #         #ecvl.Stack([img_ecvl, img_ecvl, img_ecvl], img_ecvl)
        #         #img_ecvl.channels_ = "xyc"
        #         #img_ecvl.colortype_ = ecvl.ColorType.BGR
        #         ecvl.ResizeDim(img_ecvl, img_ecvl, (512,512),
        #                            ecvl.InterpolationType.nearest)
        #         image_np = np.array(img_ecvl, copy=False)
        #         pred_np = np.array(pred_ecvl, copy=False)
        #         gt_np = np.array(gt_ecvl, copy=False)

        #         pred_np = pred_np.squeeze()
        #         gt_np = gt_np.squeeze()
        #         # Prediction summed in R channel
        #         image_np[:, :, -1] = np.where(pred_np == 255, pred_np,
        #                                         image_np[:, :, -1])
        #         # Ground truth summed in G channel
        #         image_np[:, :, 1] = np.where(gt_np == 255, gt_np,
        #                                         image_np[:, :, 1])

        #         head, tail = os.path.splitext(os.path.basename(filename))
        #         bname = "{}.png".format(head)
        #         filepath = os.path.join(args.runs_dir, "images", bname)
        #         ecvl.ImWrite(filepath, img_ecvl)
    d.Stop()
    test_time = ((time.time() - start_time) / num_batches_test) / args.batch_size
    miou = iou_evaluator.MeanMetric()
    mdice = dice_evaluator.MeanMetric()
    run.summary["test-dice"] =  mdice
    run.summary["test-iou"] = miou
    run.summary["test-time"] = test_time
    run.update()
    print("Test IoU: %.6g" % miou)
    print("Test dice: %.6g" % mdice)
    print("---Inference takes %s seconds ---" % test_time)


#python3 test_model.py ../data/processed/dataset_molinette20210418/dataset_molinette.yml --ckpts eidoslab/deephealth-uc4/1vmk2hrm/dh-uc4_epoch_2_miou_0.008275168765559426.bin --batch-size 12 --gpu 0 1 0 0
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", metavar="INPUT_DATASET",default=None)
    parser.add_argument("--ckpts", metavar='CHECKPOINTS_PATH', default='dh-uc4_epoch_63_miou_0.5952193501975717.bin')
    parser.add_argument("--wb_run_path", metavar='RUN_PATH', default='eidoslab/deephealth-uc4/3r7s9qkd')
    parser.add_argument("--batch-size", type=int, metavar="INT", default=1)
    parser.add_argument("--shape", type=int, default=512)
    parser.add_argument('--gpu', nargs='+', type=int, required=False, help='`--gpu 1 1` to use two GPUs')
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--queue_ratio_size", type=int, default=4)
    parser.add_argument("--mem", metavar="|".join(MEM_CHOICES), choices=MEM_CHOICES, default="low_mem")
    parser.add_argument("--runs-dir", metavar="DIR",
                        help="if set, save images, checkpoints and logs in this directory")
    main(parser.parse_args())
