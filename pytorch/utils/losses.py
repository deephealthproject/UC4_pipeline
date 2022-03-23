import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils.lovasz_losses import lovasz_softmax

def make_one_hot(labels, classes):
    one_hot = torch.cuda.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_()
    target = one_hot.scatter_(1, labels.data, 1)
    return target

def get_weights(target):
    t_np = target.view(-1).data.cpu().numpy()

    classes, counts = np.unique(t_np, return_counts=True)
    cls_w = np.median(counts) / counts
    #cls_w = class_weight.compute_class_weight('balanced', classes, t_np)

    weights = np.ones(7)
    weights[classes] = cls_w
    return torch.from_numpy(weights).float().cuda()

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        self.CE = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target):
        loss = self.CE(output, target)
        return loss

class BCEWithLogitsLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='none'):
        super(BCEWithLogitsLoss, self).__init__()
        #self.BCE = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([0.5]).cuda(), reduction=reduction)
        self.BCE = nn.BCEWithLogitsLoss(weight=weight, reduction=reduction)
        #self.BCE = nn.BCEWithLogitsLoss(weight=weight, reduction=reduction, pos_weight=torch.tensor([474]).cuda())

    def forward(self, output, target):
        loss = self.BCE(output.squeeze(), target.squeeze().type_as(output))
        return loss

class DiceLoss(nn.Module):
    def __init__(self, weight=None,  ignore_index=255, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, threshold=None):
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
        dice = (2.*intersection + smooth)/(inputs.sum(1) + targets.sum(1) + smooth)
        dice = 1 - dice  
        return dice

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=255, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(reduce=False, ignore_index=ignore_index, weight=alpha)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)
        loss = ((1-pt)**self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, ignore_index=255):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits

    def forward(self, output, target):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(output.squeeze(), target.squeeze().type_as(output))
        else:
            BCE_loss = F.binary_cross_entropy(output.squeeze(), target.squeeze().type_as(output))
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        return torch.mean(F_loss)

class CE_DiceLoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255, weight=None):
        super(CE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)
    
    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)
        return CE_loss + dice_loss

class BCE_DiceLoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255, weight=None):
        super(BCE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss(weight=weight, reduction=reduction)
    
    def forward(self, output, target):
        bce_loss = self.bce(output.squeeze(), target.squeeze().type_as(output))
        dice_loss = self.dice(output, target)
        return bce_loss + dice_loss

class LovaszSoftmax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore_index=255):
        super(LovaszSoftmax, self).__init__()
        self.smooth = classes
        self.per_image = per_image
        self.ignore_index = ignore_index
    
    def forward(self, output, target):
        logits = F.softmax(output, dim=1)
        loss = lovasz_softmax(logits, target, ignore=self.ignore_index)
        return loss
