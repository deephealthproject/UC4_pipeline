"""
Lung nodule segmentation training with pyecvl/pyeddl.
https://github.com/deephealthproject/pyecvl
"""

import argparse
import os
import random
import time
import wandb
import numpy as np
import pandas as pd
import pyecvl.ecvl as ecvl
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
import models.utils as utils
from models.models import UNet
from models.models import Nabla

wandb.login()
MEM_CHOICES = ("low_mem", "mid_mem", "full_mem")
ARGS = None

def main():
    args = ARGS
    wandb.init(project="deephealth-uc4")
    num_workers = wandb.config.num_workers
    queue_ratio_size = wandb.config.queue_ratio_size
    num_classes = 1
    training_time = 0
    tot_training_time = 0
    size = [args.shape, args.shape]  # size of images
    if args.shape == 512:
        mean = 0.3285
        std = 0.3556
    else:
        mean = 0
        std = 1
    thresh = 0.5
    miou_best = -1

    in_ = eddl.Input([1, size[0], size[1]])
    out = UNet(in_, num_classes)
    out_sigm = eddl.Sigmoid(out)
    net = eddl.Model([in_], [out_sigm])
    count = 0
    for layer in net.layers:
        for params in layer.params:
            count += params.size
    print("Number of trainable parameters: {}".format(count))
    print(args.gpu)
    eddl.build(
        net,
        eddl.adam(0.001),
        ["dice"],
        ["dice"],
        eddl.CS_GPU(args.gpu, mem=args.mem) if args.gpu else eddl.CS_CPU(mem=args.mem)
    )
    eddl.summary(net)

    if args.resume_ckpts and os.path.exists(args.resume_ckpts):
        print("Loading checkpoints '{}'".format(args.resume_ckpts))
        eddl.load(net, args.resume_ckpts, 'bin')

    training_augs = ecvl.SequentialAugmentationContainer([
        ecvl.AugResizeDim(size),
        ecvl.AugRotate([-10, 10])
    ])
    validation_augs = ecvl.SequentialAugmentationContainer([
        ecvl.AugResizeDim(size)
    ])
    dataset_augs = ecvl.DatasetAugmentations([
        training_augs, validation_augs, None
    ])

    print("Reading dataset")
    d = ecvl.DLDataset(args.dataset, 
                       args.batch_size, 
                       dataset_augs, 
                       ctype=ecvl.ColorType.GRAY, 
                       ctype_gt=ecvl.ColorType.GRAY, 
                       num_workers=num_workers,
                       queue_ratio_size=queue_ratio_size,
                       drop_last=[True, False, False])
    x = Tensor([args.batch_size, d.n_channels_, size[0], size[1]])
    y = Tensor([args.batch_size, d.n_channels_gt_, size[0], size[1]])
    num_samples_train = len(d.GetSplit())
    num_batches_train = num_samples_train // args.batch_size
    d.SetSplit(ecvl.SplitType.validation)
    num_samples_validation = len(d.GetSplit())
    num_batches_validation = num_samples_validation // args.batch_size
    indices = list(range(args.batch_size))

    iou_evaluator = utils.Evaluator()
    print("Starting training")
    for e in range(args.epochs):
        # TRAINING
        print("Epoch {:d}/{:d} - Training".format(e + 1, args.epochs),
              flush=True)
        d.SetSplit(ecvl.SplitType.training)
        eddl.reset_loss(net)
        s = d.GetSplit()
        random.shuffle(s)
        #d.split_.training_ = s
        d.ResetAllBatches(shuffle=True)
        start_time = time.time()
        # Spawn the threads
        d.Start()
        for i, b in enumerate(range(num_batches_train)):
            #d.LoadBatch(x, y)
            samples, x, y = d.GetBatch()
            x.div_(255.0)
            y.div_(255.0)
            x.sub_(mean)
            x.div_(std)
            tx, ty = [x], [y]
            eddl.train_batch(net, tx, ty, indices)
            if i % args.log_interval == 0:
                print("Epoch {:d}/{:d} (batch {:d}/{:d}) - ".format(
                    e + 1, args.epochs, b + 1, num_batches_train
                ), end="", flush=True)
                eddl.print_loss(net, b)
                print()
        d.Stop()
        training_time = time.time() - start_time
        wandb.log({"train-time": training_time}, commit=False) 
        tot_training_time += training_time
        print("---Training takes %s seconds ---" % training_time)
        losses1 = eddl.get_losses(net)
        metrics1 = eddl.get_metrics(net)
        for l, m in zip(losses1, metrics1):
            print("Loss: %.6f\tMetric: %.6f" % (l, m))
            wandb.log({"train-loss": l}, commit=False)
            wandb.log({"train-dice": m}, commit=False)

        # EVALUATION
        d.SetSplit(ecvl.SplitType.validation)
        # Reset current split without shuffling
        d.ResetBatch(d.current_split_, False)
        iou_evaluator.ResetEval()
        loss_evaluator = utils.Evaluator()
        loss_evaluator.ResetEval()
        print("Epoch %d/%d - Evaluation" % (e + 1, args.epochs), flush=True)
        start_time = time.time()
        d.Start()
        for b in range(num_batches_validation):
            print("Epoch {:d}/{:d} (batch {:d}/{:d}) ".format(
                e + 1, args.epochs, b + 1, num_batches_validation
            ), end="", flush=True)
            samples, x, y = d.GetBatch()
            x.div_(255.0)
            x.sub_(mean)
            x.div_(std)
            y.div_(255.0)
            eddl.forward(net, [x])
            output = eddl.getOutput(out_sigm)
            iou = iou_evaluator.BinaryIoU(np.array(output), np.array(y), thresh=thresh)
            loss = loss_evaluator.DiceCoefficient(np.array(output), np.array(y), thresh=thresh)
            print("- Batch IoU: %.6g " % iou, end="", flush=True)
            print("- Batch loss: %.6g " % loss, end="", flush=True)
            print()
        d.Stop()
        miou = iou_evaluator.MeanMetric()
        mloss = sum(loss_evaluator.buf) / len(loss_evaluator.buf)
        validation_time = time.time() - start_time
        print("Val IoU: %.6g" % miou)
        print("Val loss: %.6g" % mloss)
        wandb.log({"val-loss": mloss}, commit=False)
        wandb.log({"val-dice": 1-mloss}, commit=False)
        wandb.log({"val-iou": miou}, commit=False)
        wandb.log({"val-time": validation_time}, commit=True)
        print("---Validation takes %s seconds ---" % validation_time)
        
        if miou > miou_best:
            print("Saving weights")
            checkpoint_path = os.path.join(wandb.run.dir, 
                                           "dh-uc4_epoch_{}_miou_{}.bin".format(e+1, miou))
            eddl.save(net, checkpoint_path, "bin")
            wandb.save(os.path.join(wandb.run.dir, "*.bin"))
            miou_best = miou
    wandb.log({"total-training-time": tot_training_time, "average-training-time": tot_training_time/args.epochs})


#python3 train_model.py ../data/processed/dataset_molinette20210418/dataset_molinette.yml --epochs 100 --batch-size 8 --runs-dir ../runs/dataset_molinette20210418/UNet/ --gpu 0 1 0 0
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", metavar="INPUT_DATASET", default=None)
    parser.add_argument("--epochs", type=int, metavar="INT", default=20)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=3)
    parser.add_argument("--shape", type=int, default=512)
    parser.add_argument("--log-interval", type=int, metavar="INT", default=1)
    parser.add_argument('--gpu', nargs='+', type=int, required=False, help='`--gpu 1 1` to use two GPUs')
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--queue_ratio_size", type=int, default=1)
    parser.add_argument("--resume_ckpts", type=str)
    parser.add_argument("--mem", metavar="|".join(MEM_CHOICES), choices=MEM_CHOICES, default="low_mem")
    ARGS = parser.parse_args()
    sweep_config = {
        "name" : "sweep-2-gpus",
        "method" : "grid",
        "metric" : {
            "goal": "minimize",
            "name": "train-time"
        },
        "parameters" : {
            "num_workers" : {
            "values" : [1, 2, 4, 8, 16]
            },
            "queue_ratio_size" : {
            "values" : [1, 2, 4, 8, 16]
            }
        }
    }
    sweep_id = wandb.sweep(sweep_config, project="deephealth-uc4")
    wandb.agent(sweep_id, function=main, project="deephealth-uc4")
