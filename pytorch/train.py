import os
import json
import argparse
import torch
import wandb
import dataloaders
import models
import random
import numpy as np
from utils import losses
from utils import Logger
from trainer import Trainer

wandb.login()
wandb.init(project="deephealth-uc4", tags=["PyTorch_{}".format(torch.__version__)])

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)

def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT 
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def main(config, resume, wb_run_path):
    set_seed(config["seed"])
    train_logger = Logger()

    # DATA LOADERS
    train_loader = get_instance(dataloaders, 'train_loader', config)
    val_loader = get_instance(dataloaders, 'val_loader', config)
    test_loader = get_instance(dataloaders, 'test_loader', config)

    # MODEL
    if config["arch"]["type"] == "UNet3D":
        model = get_instance(models, 'arch', config)
    else:
        model = get_instance(models, 'arch', config, config['num_classes'], config['in_channels'])
    availble_gpus = list(range(torch.cuda.device_count()))
    print(availble_gpus)
    if resume is not None:
        #checkpoint = torch.load(resume)
        checkpoint = wandb.restore(resume, run_path=wb_run_path)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model, device_ids=availble_gpus)
        model.load_state_dict(checkpoint)
    print(f'\n{model}\n')
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)

    # LOSS
    loss = getattr(losses, config['loss'])(ignore_index = config['ignore_index'])

    # TRAINING
    trainer = Trainer(
        model=model,
        loss=loss,
        resume=None,
        wb_run_path=None,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_logger=train_logger,
        prefetch=True)

    trainer.train()

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-w', '--wb_run_path', default=None, type=str,
                        help='Weights & Biases run id path (like eidoslab/deephealth-uc4/3r7s9qkd) of the checkpoint to be resumed')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    config = json.load(open(args.config))
    #if args.resume:
    #    config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    main(config, args.resume, args.wb_run_path)