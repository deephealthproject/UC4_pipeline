import torch
import time
import numpy as np
from torchvision.utils import make_grid
from torchvision import transforms
from utils import transforms as local_transforms
from base import BaseTrainer, DataPrefetcher
from utils.helpers import colorize_mask
from utils.metrics import AverageValueMeter, eval_metrics, AverageMeter
from tqdm import tqdm
from torch.cuda.amp import autocast 

class Trainer(BaseTrainer):
    def __init__(self, model, loss, resume, config, train_loader, val_loader=None, test_loader=None, train_logger=None, prefetch=True):
        super(Trainer, self).__init__(model, loss, resume, config, train_loader, val_loader, test_loader, train_logger)
        
        self.wrt_mode, self.wrt_step = 'train_', 0
        self.log_step = config['trainer'].get('log_per_iter', int(np.sqrt(self.train_loader.batch_size)))
        if config['trainer']['log_per_iter']: self.log_step = int(self.log_step / self.train_loader.batch_size) + 1

        self.num_classes = config['num_classes']
        self.accumulation_steps = config['accumulation_steps']
        
        if self.device ==  torch.device('cpu'): prefetch = False
        if prefetch:
            self.train_loader = DataPrefetcher(train_loader, device=self.device)
            self.val_loader = DataPrefetcher(val_loader, device=self.device)
            self.test_loader = DataPrefetcher(test_loader, device=self.device)

        torch.backends.cudnn.benchmark = True

    def _train_epoch(self, epoch):
        self.logger.info('\n')
            
        self.model.train()
        if self.config['arch']['args']['freeze_bn']:
            if isinstance(self.model, torch.nn.DataParallel): self.model.module.freeze_bn()
            else: self.model.freeze_bn()
        self.wrt_mode = 'train'

        tic = time.time()
        self._reset_metrics()
        tbar = tqdm(self.train_loader, ncols=130)
        for batch_idx, (data, target) in enumerate(tbar):
            self.data_time.update(time.time() - tic)
            data, target = data.to(self.device), target.to(self.device)

            # LOSS & OPTIMIZE
            if self.config["mixed_precision"]:
                with autocast():
                    output = self.model(data)
                    if self.config['arch']['type'][:3] == 'PSP':
                        assert output[0].size()[2:] == target.size()[1:]
                        assert output[0].size()[1] == self.num_classes 
                        loss = self.loss(output[0], target)
                        loss += self.loss(output[1], target) * 0.4
                        output = output[0]
                    else:
                        assert output.size()[2:] == target.size()[1:]
                        assert output.size()[1] == self.num_classes 
                        loss = self.loss(output, target)
            else:
                output = self.model(data)
                if self.config['arch']['type'][:3] == 'PSP':
                    assert output[0].size()[2:] == target.size()[1:]
                    assert output[0].size()[1] == self.num_classes 
                    loss = self.loss(output[0], target)
                    loss += self.loss(output[1], target) * 0.4
                    output = output[0]
                else:
                    assert output.size()[2:] == target.size()[1:]
                    assert output.size()[1] == self.num_classes 
                    loss = self.loss(output, target)
            if isinstance(self.loss, torch.nn.DataParallel):
                loss = loss.mean()
            loss = loss / self.accumulation_steps
            self.total_loss.append(loss)
            if self.config["mixed_precision"]:
                self.scaler.scale(loss.mean()).backward()
            else:
                loss.mean().backward()

            is_last_step = (batch_idx+1) == len(tbar)
            do_step = (batch_idx+1) % self.accumulation_steps == 0 or is_last_step

            if do_step:   # Wait for several backward steps
                if self.config["mixed_precision"]:           
                    self.scaler.step(self.optimizer)  # Now we can do an optimizer step
                    self.scaler.update()
                    self.optimizer.zero_grad() 
                else:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # measure elapsed time
                self.batch_time.update(time.time() - tic)
                tic = time.time()
            
            # FOR EVAL
            seg_metrics = eval_metrics(self.config["task"], output, target, self.num_classes)
            self._update_seg_metrics(*seg_metrics)

            if (batch_idx+1) % self.accumulation_steps == 0:
                if self.config["task"] != "classification":
                    pixAcc, mIoU, _ = self._get_seg_metrics().values()

                    # PRINT INFO
                    tbar.set_description('TRAIN ({}) | Loss: {:.3f} | Acc {:.2f} mIoU {:.2f} | B {:.2f} D {:.2f} |'.format(
                                            epoch, torch.cat(self.total_loss, 0).mean().item(), 
                                            pixAcc, mIoU,
                                            self.batch_time.mean, self.data_time.mean))
                else:
                    m = self._get_seg_metrics()
                    acc = m["Accuracy"]
                    TP = m["TP"]
                    FN = m["FN"]
                    TN = m["TN"]
                    FP = m["FP"]
                    sens = m["Sensitivity"]
                    spec = m["Specificity"]

                    # PRINT INFO
                    tbar.set_description('TRAIN ({}) | Loss: {:.3f} | Acc: {:.2f}, TP: {:.2f}, FN: {:.2f}, TN: {:.2f}, FP: {:.2f}, Sensitivity: {:.2f}, Specificity: {:.2f} | B {:.2f} D {:.2f} |'.format(
                                            epoch, torch.cat(self.total_loss, 0).mean().item(), 
                                            acc, TP, FN, TN, FP, sens, spec, self.batch_time.mean, self.data_time.mean))

        # METRICS TO TENSORBOARD
        self.writer.add_scalar(f'{self.wrt_mode}/loss', torch.cat(self.total_loss, 0).mean().item(), epoch)
        seg_metrics = self._get_seg_metrics()
        for k, v in list(seg_metrics.items()): 
            self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, epoch)
        for i, opt_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(f'{self.wrt_mode}/Learning_rate_{i}', opt_group['lr'], epoch)

        # RETURN LOSS & METRICS
        log = {'loss': torch.cat(self.total_loss, 0).mean().item(),
                **seg_metrics}

        return log

    def _valid_epoch(self, epoch):
        if self.val_loader is None:
            self.logger.warning('Not data loader was passed for the validation step, No validation is performed !')
            return {}
        self.logger.info('\n###### VALIDATION ######')

        self.model.eval()
        self.wrt_mode = 'val'

        self._reset_metrics()
        tbar = tqdm(self.val_loader, ncols=130)
        with torch.no_grad():
            val_visual = []
            for batch_idx, (data, target) in enumerate(tbar):
                data, target = data.to(self.device), target.to(self.device)
                # LOSS
                output = self.model(data)
                loss = self.loss(output, target)
                if isinstance(self.loss, torch.nn.DataParallel):
                    loss = loss.mean()
                self.total_loss.append(loss)

                seg_metrics = eval_metrics(self.config["task"], output, target, self.num_classes)
                self._update_seg_metrics(*seg_metrics)

                if self.config["task"] != "classification":
                    pixAcc, mIoU, _ = self._get_seg_metrics().values()

                    # PRINT INFO
                    tbar.set_description('VAL ({}) | Loss: {:.3f}, PixelAcc: {:.2f}, Mean IoU: {:.2f} |'.format( epoch,
                                                torch.cat(self.total_loss, 0).mean().item(),
                                                pixAcc, mIoU))
                else:
                    m = self._get_seg_metrics()
                    acc = m["Accuracy"]
                    TP = m["TP"]
                    FN = m["FN"]
                    TN = m["TN"]
                    FP = m["FP"]
                    sens = m["Sensitivity"]
                    spec = m["Specificity"]

                    # PRINT INFO
                    tbar.set_description('VAL ({}) | Loss: {:.3f}, Acc: {:.2f}, TP: {:.2f}, FN: {:.2f}, TN: {:.2f}, FP: {:.2f}, Sensitivity: {:.2f}, Specificity: {:.2f} |'.format( epoch,
                                                torch.cat(self.total_loss, 0).mean().item(),acc, TP, FN, TN, FP, sens, spec))

            # METRICS TO TENSORBOARD
            self.wrt_step = (epoch) * len(self.val_loader)
            self.writer.add_scalar(f'{self.wrt_mode}/loss', torch.cat(self.total_loss, 0).mean().item(), epoch)
            seg_metrics = self._get_seg_metrics()
            for k, v in list(seg_metrics.items()): 
                self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, epoch)

            log = {
                'val_loss': torch.cat(self.total_loss, 0).mean().item(),
                **seg_metrics
            }

        if self.lr_scheduler is not None: 
            #self.lr_scheduler.step(self.total_loss.mean)
            self.lr_scheduler.step()

        return log
    
    def _test_epoch(self, epoch):
        if self.test_loader is None:
            self.logger.warning('Not data loader was passed for the test step, No test is performed !')
            return {}
        self.logger.info('\n###### TEST ######')

        self.model.eval()
        self.wrt_mode = 'test'

        self._reset_metrics()
        tbar = tqdm(self.test_loader, ncols=130)
        with torch.no_grad():
            test_visual = []
            for batch_idx, (data, target) in enumerate(tbar):
                data, target = data.to(self.device), target.to(self.device)
                # LOSS
                output = self.model(data)
                loss = self.loss(output, target)
                if isinstance(self.loss, torch.nn.DataParallel):
                    loss = loss.mean()
                self.total_loss.append(loss)

                seg_metrics = eval_metrics(self.config["task"], output, target, self.num_classes)
                self._update_seg_metrics(*seg_metrics)

                # PRINT INFO
                if self.config["task"] != "classification":
                    pixAcc, mIoU, _ = self._get_seg_metrics().values()

                    # PRINT INFO
                    tbar.set_description('TEST ({}) | Loss: {:.3f}, PixelAcc: {:.2f}, Mean IoU: {:.2f} |'.format(epoch,
                                                torch.cat(self.total_loss, 0).mean().item(),
                                                pixAcc, mIoU))
                else:
                    m = self._get_seg_metrics()
                    acc = m["Accuracy"]
                    TP = m["TP"]
                    FN = m["FN"]
                    TN = m["TN"]
                    FP = m["FP"]
                    sens = m["Sensitivity"]
                    spec = m["Specificity"]

                    # PRINT INFO
                    tbar.set_description('TEST ({}) | Loss: {:.3f}, Acc: {:.2f}, TP: {:.2f}, FN: {:.2f}, TN: {:.2f}, FP: {:.2f}, Sensitivity: {:.2f}, Specificity: {:.2f} |'.format( epoch,
                                                torch.cat(self.total_loss, 0).mean().item(),acc, TP, FN, TN, FP, sens, spec))

            # METRICS TO TENSORBOARD
            self.wrt_step = (epoch) * len(self.test_loader)
            self.writer.add_scalar(f'{self.wrt_mode}/loss', torch.cat(self.total_loss, 0).mean().item(), epoch)
            seg_metrics = self._get_seg_metrics()
            for k, v in list(seg_metrics.items()): 
                self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, epoch)

            log = {
                'test_loss': torch.cat(self.total_loss, 0).mean().item(),
                **seg_metrics
            }

        return log

    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = []
        self.total_correct, self.total_label = 0, 0
        self.TP, self.FN, self.TN, self.FP = 0, 0, 0, 0
        self.total_iou = []
        self.total_dice = []

    def _update_seg_metrics(self, correct, labeled, TN=0, FP=0, FN=0, TP=0, iou=None, dice=None):
        self.total_correct += correct
        self.total_label += labeled
        self.TP += TP
        self.FN += FN
        self.TN += TN
        self.FP += FP
        self.total_iou.append(iou)
        self.total_dice.append(dice)

    def _get_seg_metrics(self):

        if self.config["task"] != "classification":
            pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
            #IoU = self.total_inter / self.total_union
            #mIoU = IoU.mean
            mIoU =  torch.cat(self.total_iou, 0).mean().item()
            mDiceCoeff =  torch.cat(self.total_dice, 0).mean().item()
            #print(mIoU)
            return {
                "Pixel_Accuracy": np.round(pixAcc, 3),
                "Mean_IoU": np.round(mIoU, 3),
                "Mean_DiceCoeff":  np.round(mDiceCoeff,3)
            }
        else:
            acc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
            TP =  self.TP
            FN =  self.FN
            TN =  self.TN
            FP =  self.FP
            sens = TP/(TP+FN+1e-7) #because the confusion matrix is normalized
            spec = TN/(TN+FP+1e-7)
            g_mean = np.sqrt(sens*spec)
            return {
                "Accuracy": np.round(acc, 3),
                "TP": np.round(TP, 2),
                "FN": np.round(FN, 2),
                "TN": np.round(TN, 2),
                "FP": np.round(FP, 2),
                "Sensitivity": np.round(sens, 2),
                "Specificity": np.round(spec, 2),
                "G-mean": np.round(g_mean, 2),
            }