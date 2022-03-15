import copy
import os
import torch
import torchvision
import json
import time
import wandb
import argparse
import ops.utils as utils
import torch.nn as nn
from os.path import dirname, abspath
from torchsummary import summary
from dataloaders.load_data import get_dataloader

wandb.login()
# Opening wandb file
f = open('wandb_project.json',)
wand_settings = json.load(f)

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, checkpointsdir):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    epoch = 0
    best_epoch = 0
    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_test_acc = []
    best_test_loss = []
    while epoch - best_epoch <= config.patience:
        print('Epoch {}'.format(epoch))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test', 'validation']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'test':
                best_test_acc.append(epoch_acc)
                best_test_loss.append(epoch_loss)
            
            wandb.log({'{}/accuracy'.format(phase): epoch_acc}, step=epoch)
            wandb.log({'{}/loss'.format(phase): epoch_loss}, step=epoch)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'validation' and epoch_loss < best_val_loss:
                best_epoch = epoch
                best_val_acc = epoch_acc
                best_val_loss = epoch_loss
                wandb.run.summary["best_epoch"] = epoch
                wandb.run.summary["best_test_acc"] = best_test_acc[best_epoch]
                wandb.run.summary["best_test_loss"] = best_test_loss[best_epoch]
                wandb.run.summary["best_val_acc"] = best_val_acc
                wandb.run.summary["best_val_loss"] = best_val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                utils.save_checkpoint({
                    "epoch": best_epoch,
                    "state_dict": model.state_dict(),
                    "metric": config.monitor,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }, True, checkpointsdir, "{}_resnet_18".format(best_epoch), config.dataset=="mnist" and config.reconstruction=="None" and config.seed==42)

        print()
        epoch += 1

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_val_acc))
    print('Best val Loss: {:4f}'.format(best_val_loss))
    print('Best test Acc: {:4f}'.format(best_test_acc[best_epoch]))
    print('Best test Loss: {:4f}'.format(best_test_loss[best_epoch]))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def train_test_resnet(config):
    run = wandb.init(project=wand_settings["project"], entity=wand_settings["entity"], reinit=True)
    print(wandb.run.name)
    wandb.config.update(config)

    experiment_folder = utils.create_experiment_folder(config, wandb.run.name)

    utils.set_seed(config.seed)
    base_dir = dirname(dirname(abspath(__file__)))

    test_base_dir = base_dir + "/results/" + config.dataset + "/" + config.model + "/" + experiment_folder

    logdir = test_base_dir + "/logs/"
    checkpointsdir = test_base_dir + "/checkpoints/"
    runsdir = test_base_dir + "/runs/"
    imgdir = test_base_dir + "/images/"

    # Make model checkpoint directory
    if not os.path.exists(checkpointsdir):
        os.makedirs(checkpointsdir)

    # Make log directory
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # Make img directory
    if not os.path.exists(imgdir):
        os.makedirs(imgdir)

    # Set logger path
    utils.set_logger(os.path.join(logdir, "model.log"))

    # Get dataset loaders
    train_loader, valid_loader, test_loader = get_dataloader(config, base_dir)

    dataloaders = {'train': train_loader, 'validation': valid_loader, 'test': test_loader}
    dataset_sizes = {'train': len(train_loader.sampler), 'validation': len(valid_loader.sampler), 'test': len(test_loader.sampler)}

    # Enable GPU usage
    if config.use_cuda and torch.cuda.is_available():
        device = torch.device(config.cuda_device)
    else:
        device = torch.device("cpu")
    model_conv = torchvision.models.resnet34(pretrained=True)
    #for param in model_conv.parameters():
    #    param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    print(model_conv)
    print(num_ftrs)
    model_conv.fc = nn.Linear(num_ftrs, config.num_classes)

    model_conv = model_conv.to(device)

    summary(model_conv, input_size=(3, 32, 32))

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_conv.parameters()), lr=config.lr)
    else:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model_conv.parameters()), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, config.decay_rate)
    model_conv = train_model(model_conv, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, checkpointsdir)    


def main(config):
    for k in range(len(config.seeds)):
        config.seed = config.seeds[k]
        train_test_resnet(config)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', default=None, type=str, help='Config.json path')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    config = utils.DotDict(json.load(open(args.config)))
    main(config)