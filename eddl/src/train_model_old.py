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
import pyecvl
import pyeddl
import pyecvl.ecvl as ecvl
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
import models.utils as utils
from models.models import UNet
from models.models import Nabla

wandb.login()
wandb.init(project="deephealth-uc4", tags=["PyEDDL_{}".format(pyeddl.__version__), "PyECVL_{}".format(pyecvl.__version__)])
MEM_CHOICES = ("low_mem", "mid_mem", "full_mem")

def main(args):
    num_workers = args.num_workers
    queue_ratio_size = args.queue_ratio_size
    num_classes = 1
    #loss_f = "binary_cross_entropy"
    loss_f = "dice"
    metric_f = "dice"
    training_time = 0
    tot_training_time = 0
    size = [args.shape, args.shape]  # size of images
    if args.shape == 512:
        mean = 0.3266
        std = 0.3551
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
        eddl.adam(0.0001),
        [loss_f],
        [metric_f],
        eddl.CS_GPU(args.gpu, mem=args.mem) if args.gpu else eddl.CS_CPU(mem=args.mem)
    )
    eddl.summary(net)

    if args.resume_ckpts:
        print("Loading checkpoints '{}'".format(args.resume_ckpts))
        resume_model = wandb.restore(args.resume_ckpts, run_path=args.wb_run_path)
        print(resume_model.name)
        eddl.load(net, resume_model.name, "bin")

    training_augs = ecvl.SequentialAugmentationContainer([
        ecvl.AugResizeDim(size),
        ecvl.AugMirror(.5),
        ecvl.AugFlip(.5),
        ecvl.AugRotate([-20, 20]),
        ecvl.AugToFloat32(255, divisor_gt=255),
        ecvl.AugNormalize(mean, std),
    ])
    validation_augs = ecvl.SequentialAugmentationContainer([
        ecvl.AugResizeDim(size),
        ecvl.AugToFloat32(255, divisor_gt=255),
        ecvl.AugNormalize(mean, std),
    ])

    # Dataloader arguments [training,validation,test] 
    augs = [training_augs,validation_augs,validation_augs]

    # this yml describes splits in [test,training,validation] order
    # yml_order = [2,0,1]
    # augs = [augs[i] for i in yml_order]
    # drop_last = [drop_last[i] for i in yml_order]
    dataset_augs = ecvl.DatasetAugmentations(augs)

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
    d.SetSplit(ecvl.SplitType.test)
    num_samples_test = len(d.GetSplit())
    num_batches_test = num_samples_test // args.batch_size
    indices = list(range(args.batch_size))

    iou_evaluator = utils.Evaluator()
    print("Starting training")
    #eddl.set_mode(net, 1)
    for e in range(args.epochs):
        # TRAINING
        print("Epoch {:d}/{:d} - Training".format(e, args.epochs),
              flush=True)
        d.SetSplit(ecvl.SplitType.training)
        d.ResetBatch(ecvl.SplitType.training, shuffle=True)
        start_time = time.time()
        # Spawn the threads
        d.Start()
        eddl.reset_loss(net)
        for i, b in enumerate(range(num_batches_train)):
            #d.LoadBatch(x, y)
            _, x, y = d.GetBatch()
            tx, ty = [x], [y]
            eddl.train_batch(net, tx, ty, indices)
            if i % args.log_interval == 0:
                print("Epoch {:d}/{:d} (batch {:d}/{:d}) - ".format(
                    e, args.epochs, b + 1, num_batches_train
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
        print("Starting validation")
        d.SetSplit(ecvl.SplitType.validation)
        # Reset current split without shuffling
        d.ResetBatch(d.current_split_, False)
        iou_evaluator.ResetEval()
        loss_evaluator = utils.Evaluator()
        loss_evaluator.ResetEval()
        dice_evaluator = utils.Evaluator()
        dice_evaluator.ResetEval()
        print("Epoch %d/%d - Evaluation" % (e, args.epochs), flush=True)
        start_time = time.time()

        eddl.set_mode(net, 0)
        d.Start()
        for b in range(num_batches_validation):
            print("Epoch {:d}/{:d} (batch {:d}/{:d}) ".format(
                e, args.epochs, b + 1, num_batches_validation
            ), end="", flush=True)
            x, y = d.GetBatch()
            #eddl.eval_batch(net,[x],[y])
            #output = eddl.getOutput(net.lout[0])
            eddl.forward(net, [x])
            output = eddl.getOutput(out)
            iou = iou_evaluator.BinaryIoU(np.array(output), np.array(y), thresh=thresh)
            loss = loss_evaluator.DiceLoss(np.array(output), np.array(y))
            dice = dice_evaluator.DiceCoefficient(np.array(output), np.array(y), thresh=thresh)
            print("- Batch IoU: %.6g " % iou, end="", flush=True)
            print("- Batch dice: %.6g " % dice, end="", flush=True)
            print("- Batch loss: %.6g " % loss, end="", flush=True)
            print()
        d.Stop()
        miou = iou_evaluator.MeanMetric()
        mdice = dice_evaluator.MeanMetric()
        mloss = loss_evaluator.MeanMetric()
        validation_time = time.time() - start_time
        print("Val IoU: %.6g" % miou)
        print("Val dice: %.6g" % dice)
        print("Val loss: %.6g" % mloss)
        wandb.log({"validation-loss": mloss}, commit=False)
        wandb.log({"validation-dice": dice}, commit=False)
        wandb.log({"validation-iou": miou}, commit=False)
        wandb.log({"validation-time": validation_time}, commit=False)
        print("---Validation takes %s seconds ---" % validation_time)
        
        if miou > miou_best:
            print("Saving weights")
            checkpoint_path = os.path.join(wandb.run.dir, 
                                           "dh-uc4_epoch_{}_miou_{}.bin".format(e, miou))
            eddl.save(net, checkpoint_path, "bin")
            wandb.save(os.path.join(wandb.run.dir, "*.bin"))
            miou_best = miou

        # # Test
        # print("Starting testing")
        # d.SetSplit(ecvl.SplitType.test)
        # # Reset current split without shuffling
        # d.ResetBatch(d.current_split_, False)
        # iou_evaluator.ResetEval()
        # loss_evaluator = utils.Evaluator()
        # loss_evaluator.ResetEval()
        # dice_evaluator = utils.Evaluator()
        # dice_evaluator.ResetEval()
        # print("Epoch %d/%d - Test" % (e, args.epochs), flush=True)
        # start_time = time.time()
        # d.Start()
        # eddl.set_mode(net, 0)
        # for b in range(num_batches_test):
        #     print("Epoch {:d}/{:d} (batch {:d}/{:d}) ".format(
        #         e, args.epochs, b + 1, num_batches_test
        #     ), end="", flush=True)
        #     samples, x, y = d.GetBatch()
        #     eddl.forward(net, [x])
        #     output = eddl.getOutput(out_sigm)
        #     iou = iou_evaluator.BinaryIoU(np.array(output), np.array(y), thresh=thresh)
        #     loss = loss_evaluator.DiceLoss(np.array(output), np.array(y))
        #     dice = dice_evaluator.DiceCoefficient(np.array(output), np.array(y), thresh=thresh)
        #     print("- Batch IoU: %.6g " % iou, end="", flush=True)
        #     print("- Batch dice: %.6g " % dice, end="", flush=True)
        #     print("- Batch loss: %.6g " % loss, end="", flush=True)
        #     print()
        # d.Stop()
        # miou = iou_evaluator.MeanMetric()
        # mdice = dice_evaluator.MeanMetric()
        # mloss = loss_evaluator.MeanMetric()
        # validation_time = time.time() - start_time
        # print("Test IoU: %.6g" % miou)
        # print("Test dice: %.6g" % mdice)
        # print("Test loss: %.6g" % mloss)
        # wandb.log({"test-loss": mloss}, commit=False)
        # wandb.log({"test-dice": mdice}, commit=False)
        # wandb.log({"test-iou": miou}, commit=False)
        # wandb.log({"test-time": validation_time}, commit=True)
        # print("---Testing takes %s seconds ---" % validation_time)
    wandb.log({"total-training-time": tot_training_time, "average-training-time": tot_training_time/args.epochs})


#python3 train_model.py ../data/processed/dataset_molinette20210418/dataset_molinette.yml --epochs 100 --batch-size 8 --runs-dir ../runs/dataset_molinette20210418/UNet/ --gpu 0 1 0 0
#riccardo@latitude-5300:~/projects/UC4_pipeline$ sudo docker build  -t riccardorenzulli/deephealth-uc4:cudnn .
#sudo docker push riccardorenzulli/deephealth-uc4:cudnn
#kubectl apply -f eddl/pod-deephealth-uc4-training-4gpu.yaml
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", metavar="INPUT_DATASET", default=None)
    parser.add_argument("--epochs", type=int, metavar="INT", default=20)
    parser.add_argument("--batch_size", type=int, metavar="INT", default=3)
    parser.add_argument("--shape", type=int, default=512)
    parser.add_argument("--log-interval", type=int, metavar="INT", default=1)
    parser.add_argument('--gpu', nargs='+', type=int, required=False, help='`--gpu 1 1` to use two GPUs')
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--queue_ratio_size", type=int, default=1)
    parser.add_argument("--wb_run_path", metavar='RUN_PATH', default='eidoslab/deephealth-uc4/3r7s9qkd')
    parser.add_argument("--resume_ckpts", type=str)
    parser.add_argument("--mem", metavar="|".join(MEM_CHOICES), choices=MEM_CHOICES, default="low_mem")
    main(parser.parse_args())
