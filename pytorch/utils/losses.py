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
        #print(intersection.size())                            
        dice = (2.*intersection + smooth)/(inputs.sum(1) + targets.sum(1) + smooth)
        dice = 1 - dice  
        #print(dice.size())
        return dice


# class _AbstractDiceLoss(nn.Module):
#     """
#     Base class for different implementations of Dice loss.
#     """

#     def __init__(self, weight=None, normalization='sigmoid'):
#         super(_AbstractDiceLoss, self).__init__()
#         self.register_buffer('weight', weight)
#         # The output from the network during training is assumed to be un-normalized probabilities and we would
#         # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
#         # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
#         # However if one would like to apply Softmax in order to get the proper probability distribution from the
#         # output, just specify `normalization=Softmax`
#         assert normalization in ['sigmoid', 'softmax', 'none']
#         if normalization == 'sigmoid':
#             self.normalization = nn.Sigmoid()
#         elif normalization == 'softmax':
#             self.normalization = nn.Softmax(dim=1)
#         else:
#             self.normalization = lambda x: x

#     def dice(self, input, target, weight):
#         # actual Dice score computation; to be implemented by the subclass
#         raise NotImplementedError

#     def forward(self, input, target):
#         # get probabilities from logits
#         input = self.normalization(input)

#         # compute per channel Dice coefficient
#         per_channel_dice = self.dice(input, target, weight=self.weight)

#         # average Dice score across all channels/classes
#         return 1. - torch.mean(per_channel_dice)

# def flatten(tensor):
#     """Flattens a given tensor such that the channel axis is first.
#     The shapes are transformed as follows:
#        (N, C, D, H, W) -> (C, N * D * H * W)
#     """
#     # number of channels
#     C = tensor.size(1)
#     # new axis order
#     axis_order = (1, 0) + tuple(range(2, tensor.dim()))
#     # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
#     transposed = tensor.permute(axis_order)
#     # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
#     return transposed.contiguous().view(C, -1)

# def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
#     """
#     Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
#     Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.
#     Args:
#          input (torch.Tensor): NxCxSpatial input tensor
#          target (torch.Tensor): NxCxSpatial target tensor
#          epsilon (float): prevents division by zero
#          weight (torch.Tensor): Cx1 tensor of weight per channel/class
#     """

#     # input and target shapes must match
#     assert input.size() == target.size(), "'input' and 'target' must have the same shape"

#     input = flatten(input)
#     target = flatten(target)
#     target = target.float()

#     # compute per channel Dice Coefficient
#     intersect = (input * target).sum(-1)
#     if weight is not None:
#         intersect = weight * intersect

#     # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
#     denominator = (input * input).sum(-1) + (target * target).sum(-1)
#     return 2 * (intersect / denominator.clamp(min=epsilon))

# class DiceLoss(_AbstractDiceLoss):
#     """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
#     For multi-class segmentation `weight` parameter can be used to assign different weights per class.
#     The input to the loss function is assumed to be a logit and will be normalized by the Sigmoid function.
#     """

#     def __init__(self, weight=None, normalization='sigmoid'):
#         super().__init__(weight, normalization)

#     def dice(self, input, target, weight):
#         return compute_per_channel_dice(input, target, weight=self.weight)


# class GeneralizedDiceLoss(_AbstractDiceLoss):
#     """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf.
#     """

#     def __init__(self, weight=None,  ignore_index=255, size_average=True, normalization='sigmoid', epsilon=1e-6):
#         super().__init__(weight=None, normalization=normalization)
#         self.epsilon = epsilon

#     def dice(self, input, target, weight):
#         target = torch.unsqueeze(target, 1)
#         assert input.size() == target.size(), "'input' and 'target' must have the same shape"

#         input = flatten(input)
#         target = flatten(target)
#         target = target.float()

#         if input.size(0) == 1:
#             # for GDL to make sense we need at least 2 channels (see https://arxiv.org/pdf/1707.03237.pdf)
#             # put foreground and background voxels in separate channels
#             input = torch.cat((input, 1 - input), dim=0)
#             target = torch.cat((target, 1 - target), dim=0)

#         # GDL weighting: the contribution of each label is corrected by the inverse of its volume
#         w_l = target.sum(-1)
#         w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
#         w_l.requires_grad = False

#         intersect = (input * target).sum(-1)
#         intersect = intersect * w_l

#         denominator = (input + target).sum(-1)
#         denominator = (denominator * w_l).clamp(min=self.epsilon)

#         return 2 * (intersect.sum() / denominator.sum())

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
