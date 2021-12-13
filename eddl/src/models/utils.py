import numpy as np

class Evaluator:
    def __init__(self):
        self.eps = 1e-06
        self.buf = []

    def ResetEval(self):
        self.buf = []

    def BinaryIoU(self, a, b, thresh=0.5):
        intersection = np.logical_and(a > thresh, b > thresh).sum()
        union = np.logical_or(a > thresh, b > thresh).sum()
        rval = (intersection + self.eps) / (union + self.eps)
        self.buf.append(rval)
        return rval

    def DiceCoefficient(self, a, b, thresh=0.5):
        a = Threshold(a, thresh)
        b = Threshold(b, thresh)
        intersection = np.logical_and(a, b).sum()
        rval = (2 * intersection + self.eps) / (a.sum() + b.sum() + self.eps)

        self.buf.append(rval)
        return rval
    def BinaryCrossEntropy(self, y, y_pred):
        # each example is associated with a single class; sum the negative log
        # probability of the correct label over all samples in the batch.
        # observe that we are taking advantage of the fact that y is one-hot
        # encoded
        rval = -(np.sum(y * np.log(y_pred + self.eps)))/(y.shape[0]*y.shape[1]*y.shape[2]*y.shape[3])
        self.buf.append(rval)
        return rval

    def MIoU(self):
        if not self.buf:
            return 0
        return sum(self.buf) / len(self.buf)

    MeanMetric = MIoU


def Threshold(a, thresh=0.5):
    a[a > thresh] = 1
    a[a <= thresh] = 0
    return a


def ImageSqueeze(img):
    k = img.dims_.index(1)
    img.dims_ = [_ for i, _ in enumerate(img.dims_) if i != k]
    img.strides_ = [_ for i, _ in enumerate(img.strides_) if i != k]
    k = img.channels_.find("z")
    img.channels_ = "".join([_ for i, _ in enumerate(img.channels_) if i != k])