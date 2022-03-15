import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F
import torch.nn as nn
import segmentation_models_pytorch as smp

metric_fn = smp.utils.metrics.IoU(threshold=0.5, activation="sigmoid")

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = np.multiply(val, weight)
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum = np.add(self.sum, np.multiply(val, weight))
        self.count = self.count + weight
        self.avg = self.sum / self.count

    @property
    def value(self):
        return self.val

    @property
    def mean(self):
        return np.round(self.avg, 5)

class Meter(object):
    '''Meters provide a way to keep track of important statistics in an online manner.
    This class is abstract, but provides a standard interface for all meters to follow.
    '''

    def reset(self):
        '''Resets the meter to default settings.'''
        pass

    def add(self, value):
        '''Log a new value to the meter
        Args:
            value: Next result to include.
        '''
        pass

    def value(self):
        '''Get the value of the meter in the current state.'''
        pass

class AverageValueMeter(Meter):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0

    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean = 0.0 + self.sum  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan

def batch_pix_accuracy(output, target):
    _, predict = torch.max(output, 1)

    predict = predict.int() + 1
    target = target.int() + 1

    pixel_labeled = (target > 0).sum()
    pixel_correct = ((predict == target)*(target > 0)).sum()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy()

def batch_intersection_union(output, target, num_class, threshold=0.50, eps=1e-6):

    # _, predict = torch.max(output.data, 1)
    # predict = predict + 1
    # target = target + 1

    # labeled = (target > 0) * (target <= num_class)

    # predict = predict * labeled.long()
    # intersection = predict * (predict == target).long()

    # area_inter = torch.histc(intersection.float(), bins=num_class, max=num_class, min=1)
    # area_pred = torch.histc(predict.float(), bins=num_class, max=num_class, min=1)
    # area_lab = torch.histc(target.float(), bins=num_class, max=num_class, min=1)
    # area_union = area_pred + area_lab - area_inter
    # assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"
    # return ((area_inter + eps) / (area_union + eps)).cpu().numpy()

    output = output.squeeze(1)
    batch_size = output.size(0)
    output = torch.sigmoid(output)
    prediction = output > threshold
    prediction = prediction.view(batch_size,-1)
    target = target.view(batch_size,-1)
    prediction_b = prediction.byte()
    target_b = target.byte()
    intersection = (prediction_b & target_b).float().sum(1)
    union = (prediction_b | target_b).float().sum(1) 
    iou_nodule = (intersection + eps) / (union + eps)
    #print(iou_nodule.size())
    #iou_nodule = torch.mean(iou_nodule)

    # prediction = prediction + 1
    # target = target + 1
    # prediction[prediction==2] = 0
    # target[target==2] = 0
    # prediction_b = prediction.byte()
    # target_b = target.byte()
    # intersection = (prediction_b & target_b).float().sum(1)
    # union = (prediction_b | target_b).float().sum(1) 
    # iou_background = (intersection + eps) / (union + eps)
    #thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10
    #print(iou)
    #iou = (iou_nodule + iou_background) / 2
    return iou_nodule
    #return (intersection+eps).cpu().numpy(), (union+eps).cpu().numpy()

def batch_dice_coeff(inputs, targets, smooth=1, threshold=None):
    inputs = torch.sigmoid(inputs)
    if threshold is not None:
        inputs = (inputs > threshold).type(targets.dtype)
    batch_size = inputs.size(0)
    
    #flatten label and prediction tensors
    inputs = inputs.view(batch_size,-1)
    targets = targets.view(batch_size,-1)
    # input and target shapes must match
    assert inputs.size() == targets.size(), "'input' and 'target' must have the same shape"
    
    intersection = (inputs * targets).sum(1)
    #print(intersection.size())                            
    dice = (2.*intersection + smooth)/(inputs.sum(1) + targets.sum(1) + smooth)
    #print(dice.size())
    return dice

def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold)
    else:
        return x

def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)
    cm = torch.zeros(2,2)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()
    cm[0,0] = true_negatives
    cm[1,0] = false_negatives
    cm[1,1] = true_positives
    cm[0,1] = false_positives
    cm = cm + 1e-7
    cm = cm / cm.sum(axis=1, keepdims=True)
    return cm

def get_confusion_matrix(target, output):
    TN, FP, FN, TP = 0, 0, 0, 0
    for i in range(len(target)):
        if target[i] == 1 and target[i] == output[i]:
            TP += 1
        if target[i] == 0 and target[i] == output[i]:
            TN += 1
        if target[i] == 1 and target[i] != output[i]:
            FN += 1
        if target[i] == 0 and target[i] != output[i]:
            FP += 1
    return TN, FP, FN, TP

def eval_metrics(task, output, target, num_classes):
    if task != "classification":
        correct, labeled = batch_pix_accuracy(output, target)
        #iou = metric_fn(output, target.unsqueeze(1)).cpu().numpy()
        # print(iou)
        #print(output.size(), target.size())
        iou = batch_intersection_union(output, target, num_classes)
        dice = batch_dice_coeff(output, target, threshold=0.5)
        return [np.round(correct, 5), np.round(labeled, 5), iou, dice]
    else:
        output = torch.sigmoid(output)
        output = output > 0.5
        #cm = confusion(output, target).cpu().detach().numpy()
        TN, FP, FN, TP = get_confusion_matrix(target.cpu().detach().numpy(), output.cpu().detach().numpy())
        correct = output.eq(target.view_as(output)).cpu().sum().item()
        labeled = target.size(0)
        return [np.round(correct, 5), np.round(labeled, 5), TN, FP, FN, TP]

def pixel_accuracy(output, target):
    output = np.asarray(output)
    target = np.asarray(target)
    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((output == target) * (target > 0))
    return pixel_correct, pixel_labeled

def inter_over_union(output, target, num_class):
    output = np.asarray(output) + 1
    target = np.asarray(target) + 1
    output = output * (target > 0)

    intersection = output * (output == target)
    area_inter, _ = np.histogram(intersection, bins=num_class, range=(1, num_class))
    area_pred, _ = np.histogram(output, bins=num_class, range=(1, num_class))
    area_lab, _ = np.histogram(target, bins=num_class, range=(1, num_class))
    area_union = area_pred + area_lab - area_inter
    return area_inter, area_union