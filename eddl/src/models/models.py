# Copyright (c) 2020 CRS4
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

import pyeddl.eddl as eddl


def LeNet(in_layer, num_classes):
    x = in_layer
    x = eddl.MaxPool(eddl.ReLu(eddl.Conv(x, 20, [5, 5])), [2, 2], [2, 2])
    x = eddl.MaxPool(eddl.ReLu(eddl.Conv(x, 50, [5, 5])), [2, 2], [2, 2])
    x = eddl.Reshape(x, [-1])
    x = eddl.ReLu(eddl.Dense(x, 500))
    x = eddl.Softmax(eddl.Dense(x, num_classes))
    return x


def VGG16(in_layer, num_classes):
    x = in_layer
    x = eddl.ReLu(eddl.Conv(x, 64, [3, 3]))
    x = eddl.MaxPool(eddl.ReLu(eddl.Conv(x, 64, [3, 3])), [2, 2], [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 128, [3, 3]))
    x = eddl.MaxPool(eddl.ReLu(eddl.Conv(x, 128, [3, 3])), [2, 2], [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 256, [3, 3]))
    x = eddl.ReLu(eddl.Conv(x, 256, [3, 3]))
    x = eddl.MaxPool(eddl.ReLu(eddl.Conv(x, 256, [3, 3])), [2, 2], [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3]))
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3]))
    x = eddl.MaxPool(eddl.ReLu(eddl.Conv(x, 512, [3, 3])), [2, 2], [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3]))
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3]))
    x = eddl.MaxPool(eddl.ReLu(eddl.Conv(x, 512, [3, 3])), [2, 2], [2, 2])
    x = eddl.Reshape(x, [-1])
    x = eddl.ReLu(eddl.Dense(x, 4096))
    x = eddl.ReLu(eddl.Dense(x, 4096))
    x = eddl.Softmax(eddl.Dense(x, num_classes))
    return x


def SegNet(in_layer, num_classes):
    x = in_layer
    x = eddl.ReLu(eddl.Conv(x, 64, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 64, [3, 3], [1, 1], "same"))
    x = eddl.MaxPool(x, [2, 2], [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 128, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 128, [3, 3], [1, 1], "same"))
    x = eddl.MaxPool(x, [2, 2], [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 256, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 256, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 256, [3, 3], [1, 1], "same"))
    x = eddl.MaxPool(x, [2, 2], [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3], [1, 1], "same"))
    x = eddl.MaxPool(x, [2, 2], [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3], [1, 1], "same"))
    x = eddl.MaxPool(x, [2, 2], [2, 2])
    x = eddl.UpSampling(x, [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3], [1, 1], "same"))
    x = eddl.UpSampling(x, [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 256, [3, 3], [1, 1], "same"))
    x = eddl.UpSampling(x, [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 256, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 256, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 128, [3, 3], [1, 1], "same"))
    x = eddl.UpSampling(x, [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 128, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 64, [3, 3], [1, 1], "same"))
    x = eddl.UpSampling(x, [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 64, [3, 3], [1, 1], "same"))
    x = eddl.Conv(x, num_classes, [3, 3], [1, 1], "same")
    return x


def SegNetBN(x, num_classes):
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 64, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 64, [3, 3], [1, 1], "same")))
    x = eddl.MaxPool(x, [2, 2], [2, 2])
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 128, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 128, [3, 3], [1, 1], "same")))
    x = eddl.MaxPool(x, [2, 2], [2, 2])
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 256, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 256, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 256, [3, 3], [1, 1], "same")))
    x = eddl.MaxPool(x, [2, 2], [2, 2])
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 512, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 512, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 512, [3, 3], [1, 1], "same")))
    x = eddl.MaxPool(x, [2, 2], [2, 2])
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 512, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 512, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 512, [3, 3], [1, 1], "same")))
    x = eddl.MaxPool(x, [2, 2], [2, 2])

    x = eddl.UpSampling(x, [2, 2])
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 512, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 512, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 512, [3, 3], [1, 1], "same")))
    x = eddl.UpSampling(x, [2, 2])
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 512, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 512, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 256, [3, 3], [1, 1], "same")))
    x = eddl.UpSampling(x, [2, 2])
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 256, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 256, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 128, [3, 3], [1, 1], "same")))
    x = eddl.UpSampling(x, [2, 2])
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 128, [3, 3], [1, 1], "same")))
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 64, [3, 3], [1, 1], "same")))
    x = eddl.UpSampling(x, [2, 2])
    x = eddl.ReLu(
        eddl.BatchNormalization(eddl.Conv(x, 64, [3, 3], [1, 1], "same")))
    x = eddl.Conv(x, num_classes, [3, 3], [1, 1], "same")

    return x

USE_CONCAT = 1

def UNet2(x, num_classes):
    depth = 64

    x = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x, depth, [3, 3], [1, 1], "same"), True))
    x = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x, depth, [3, 3], [1, 1], "same"), True))
    x = eddl.MaxPool(x, [2, 2], [2, 2])
    x2 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x, 2*depth, [3, 3], [1, 1], "same"), True))
    x2 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x2, 2*depth, [3, 3], [1, 1], "same"), True))
    x2 = eddl.MaxPool(x2, [2, 2], [2, 2])
    x3 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x2, 4*depth, [3, 3], [1, 1], "same"), True))
    x3 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x3, 4*depth, [3, 3], [1, 1], "same"), True))
    x3 = eddl.MaxPool(x3, [2, 2], [2, 2])
    x4 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x3, 8*depth, [3, 3], [1, 1], "same"), True))
    x4 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x4, 8*depth, [3, 3], [1, 1], "same"), True))
    x4 = eddl.MaxPool(x4, [2, 2], [2, 2])

    # middle conv
    x5 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x4, 16*depth, [3, 3], [1, 1], "same"), True))
    x5 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x5, 16*depth, [3, 3], [1, 1], "same"), True))

    # decoder
    x5 = eddl.ConvT(x5, 8*depth, [2, 2], strides=[2, 2], output_padding=[0,0])

    x4_up = eddl.Concat([x4, x5]) if USE_CONCAT else eddl.Sum([x4, x5])
    x4_up = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x4_up, 8*depth, [3, 3], [1, 1], "same"), True))
    x4_up = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x4_up, 8*depth, [3, 3], [1, 1], "same"), True))
    x4_up = eddl.ConvT(x4_up, 4*depth, [2, 2], strides=[2, 2], output_padding=[0,0])

    x3_up = eddl.Concat([x3, x4_up]) if USE_CONCAT else eddl.Sum([x3, x4])
    x3_up = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x3_up, 4*depth, [3, 3], [1, 1], "same"), True))
    x3_up= eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x3_up, 4*depth, [3, 3], [1, 1], "same"), True))
    x3_up = eddl.ConvT(x3_up, 1*depth, [2, 2], strides=[2, 2], output_padding=[0,0])

    x2_up = eddl.Concat([x2, x3_up]) if USE_CONCAT else eddl.Sum([x2, x3])
    x2_up = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x2_up, 2*depth, [3, 3], [1, 1], "same"), True))
    x2_up = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x2_up, 2*depth, [3, 3], [1, 1], "same"), True))
    x2_up = eddl.ConvT(x2_up, depth, [2, 2], strides=[2, 2], output_padding=[0,0])

    x_up = eddl.Concat([x, x2_up]) if USE_CONCAT else eddl.Sum([x, x2])
    x_up = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x_up, depth, [3, 3], [1, 1], "same"), True))
    x_up = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x_up, depth, [3, 3], [1, 1], "same"), True))

    # final conv
    x_up = eddl.Conv(x_up, num_classes, [1, 1])

    return x
    
def UNet(x, num_classes):
    depth = 64

    # encoder
    x = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x, depth, [3, 3], [1, 1], "same"), True))
    x = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x, depth, [3, 3], [1, 1], "same"), True))
    x2 = eddl.MaxPool(x, [2, 2], [2, 2])#eddl.MaxPool(x, [2, 2], [2, 2])#
    x2 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x2, 2*depth, [3, 3], [1, 1], "same"), True))
    x2 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x2, 2*depth, [3, 3], [1, 1], "same"), True))
    x3 = eddl.MaxPool(x2, [2, 2], [2, 2])#eddl.MaxPool(x2, [2, 2], [2, 2])#
    x3 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x3, 4*depth, [3, 3], [1, 1], "same"), True))
    x3 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x3, 4*depth, [3, 3], [1, 1], "same"), True))
    x4 = eddl.MaxPool(x3, [2, 2], [2, 2])#eddl.MaxPool(x3, [2, 2], [2, 2])#
    x4 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x4, 8*depth, [3, 3], [1, 1], "same"), True))
    x4 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x4, 8*depth, [3, 3], [1, 1], "same"), True))
    x5 = eddl.MaxPool(x4, [2, 2], [2, 2])#eddl.MaxPool(x4, [2, 2], [2, 2])#

    # middle conv
    x5 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x5, 16*depth, [3, 3], [1, 1], "same"), True))
    x5 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x5, 16*depth, [3, 3], [1, 1], "same"), True))

    # decoder
    x5 = eddl.UpSampling(x5, [2,2])
    #x5 = eddl.Conv(x5, 8*depth, [2, 2], [1, 1], "same")
    x5 = eddl.Pad(x5, [0, 1, 1, 0]) #<-- pad fix
    x5 = eddl.Conv(x5, 8*depth, [2, 2], [1, 1], "valid")

    x4 = eddl.Concat([x4, x5])
    x4 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x4, 8*depth, [3, 3], [1, 1], "same"), True))
    x4 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x4, 8*depth, [3, 3], [1, 1], "same"), True))
    x4 = eddl.UpSampling(x4, [2,2])
    #x4 = eddl.Conv(x4, 4*depth, [2, 2], [1, 1], "same")
    x4 = eddl.Pad(x4, [0, 1, 1, 0]) #<-- pad fix
    x4 = eddl.Conv(x4, 4*depth, [2, 2], [1, 1], "valid")
    
    x3 = eddl.Concat([x3, x4])
    x3 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x3, 4*depth, [3, 3], [1, 1], "same"), True))
    x3 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x3, 4*depth, [3, 3], [1, 1], "same"), True))
    x3 = eddl.UpSampling(x3, [2,2])
    #x3 = eddl.Conv(x3, 2*depth, [2, 2], [1, 1], "same")
    x3 = eddl.Pad(x3, [0, 1, 1, 0]) #<-- pad fix
    x3 = eddl.Conv(x3, 2*depth, [2, 2], [1, 1], "valid")
    
    x2 = eddl.Concat([x2, x3])
    x2 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x2, 2*depth, [3, 3], [1, 1], "same"), True))
    x2 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x2, 2*depth, [3, 3], [1, 1], "same"), True))
    x2 = eddl.UpSampling(x2, [2,2])
    #x2 = eddl.Conv(x2, depth, [2, 2], [1, 1], "same")
    x2 = eddl.Pad(x2, [0, 1, 1, 0]) #<-- pad fix
    x2 = eddl.Conv(x2, depth, [2, 2], [1, 1], "valid")

    x = eddl.Concat([x, x2])
    x = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x, depth, [3, 3], [1, 1], "same"), True))
    x = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x, depth, [3, 3], [1, 1], "same"), True))

    # final conv
    x = eddl.Conv(x, num_classes, [1, 1])
    return x

def UNet_old(x, num_classes):
    depth = 64

    # encoder
    x = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x, depth, [3, 3], [1, 1], "same"), True))
    x = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x, depth, [3, 3], [1, 1], "same"), True))
    x2 = eddl.MaxPool(x, [2, 2], [2, 2])
    x2 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x2, 2*depth, [3, 3], [1, 1], "same"), True))
    x2 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x2, 2*depth, [3, 3], [1, 1], "same"), True))
    x3 = eddl.MaxPool(x2, [2, 2], [2, 2])
    x3 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x3, 4*depth, [3, 3], [1, 1], "same"), True))
    x3 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x3, 4*depth, [3, 3], [1, 1], "same"), True))
    x4 = eddl.MaxPool(x3, [2, 2], [2, 2])
    x4 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x4, 8*depth, [3, 3], [1, 1], "same"), True))
    x4 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x4, 8*depth, [3, 3], [1, 1], "same"), True))
    x5 = eddl.MaxPool(x4, [2, 2], [2, 2])

    # middle conv
    x5 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x5, 16*depth, [3, 3], [1, 1], "same"), True))
    x5 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x5, 16*depth, [3, 3], [1, 1], "same"), True))

    # decoder
    x5 = eddl.Conv(
        eddl.UpSampling(x5, [2, 2]), 8*depth, [2, 2], [1, 1], "same"
    )
    x4 = eddl.Concat([x4, x5])
    x4 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x4, 8*depth, [3, 3], [1, 1], "same"), True))
    x4 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x4, 8*depth, [3, 3], [1, 1], "same"), True))
    x4 = eddl.Conv(
        eddl.UpSampling(x4, [2, 2]), 4*depth, [2, 2], [1, 1], "same"
    )
    x3 = eddl.Concat([x3, x4])
    x3 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x3, 4*depth, [3, 3], [1, 1], "same"), True))
    x3 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x3, 4*depth, [3, 3], [1, 1], "same"), True))
    x3 = eddl.Conv(
        eddl.UpSampling(x3, [2, 2]), 2*depth, [2, 2], [1, 1], "same"
    )
    x2 = eddl.Concat([x2, x3])
    x2 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x2, 2*depth, [3, 3], [1, 1], "same"), True))
    x2 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x2, 2*depth, [3, 3], [1, 1], "same"), True))
    x2 = eddl.Conv(
        eddl.UpSampling(x2, [2, 2]), depth, [2, 2], [1, 1], "same"
    )
    x = eddl.Concat([x, x2])
    x = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x, depth, [3, 3], [1, 1], "same"), True))
    x = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x, depth, [3, 3], [1, 1], "same"), True))

    # final conv
    x = eddl.Conv(x, num_classes, [1, 1])

    return x

def Nabla(x, num_classes):
    # encoder
    x = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x, 64, [3, 3], [1, 1], "same"), True))
    x1 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x, 64, [3, 3], [1, 1], "same"), True))
    x = eddl.MaxPool(x1, [2, 2], [2, 2])
    x = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x, 64, [3, 3], [1, 1], "same"), True))
    x = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x, 64, [3, 3], [1, 1], "same"), True))
    x = eddl.MaxPool(x, [2, 2], [2, 2])
    x = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x, 64, [3, 3], [1, 1], "same"), True))
    x = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x, 64, [3, 3], [1, 1], "same"), True))
    x = eddl.MaxPool(x, [2, 2], [2, 2])
    x = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x, 64, [3, 3], [1, 1], "same"), True))
    x = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x, 64, [3, 3], [1, 1], "same"), True))

    # decoder
    x = eddl.BatchNormalization(eddl.Conv(x, 64, [3, 3], [1, 1], "same"), True)
    x = eddl.BatchNormalization(eddl.Conv(x, 64, [3, 3], [1, 1], "same"), True)
    x = eddl.UpSampling(x, [2, 2])  # should be unpooling
    x = eddl.BatchNormalization(eddl.Conv(x, 64, [3, 3], [1, 1], "same"), True)
    x = eddl.BatchNormalization(eddl.Conv(x, 64, [3, 3], [1, 1], "same"), True)
    x = eddl.UpSampling(x, [2, 2])  # should be unpooling
    x = eddl.BatchNormalization(eddl.Conv(x, 64, [3, 3], [1, 1], "same"), True)
    x = eddl.BatchNormalization(eddl.Conv(x, 64, [3, 3], [1, 1], "same"), True)
    x2 = eddl.UpSampling(x, [2, 2])  # should be unpooling

    # merge
    x = eddl.Concat([x1, x2])

    x = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x, 64, [3, 3], [1, 1], "same"), True))
    x = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x, 64, [3, 3], [1, 1], "same"), True))
    x = eddl.Conv(x, num_classes, [1, 1])
    #x = eddl.Sigmoid(x)

    return x

def UNet_UC3(x, num_classes):
    depth = 64

    # encoder
    x = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x, depth, [3, 3], [1, 1], "same"), True))
    x = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x, depth, [3, 3], [1, 1], "same"), True))
    x2 = eddl.MaxPool(x, [2, 2], [2, 2])#eddl.MaxPool(x, [2, 2], [2, 2])#
    x2 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x2, 2*depth, [3, 3], [1, 1], "same"), True))
    x2 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x2, 2*depth, [3, 3], [1, 1], "same"), True))
    x3 = eddl.MaxPool(x2, [2, 2], [2, 2])#eddl.MaxPool(x2, [2, 2], [2, 2])#
    x3 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x3, 4*depth, [3, 3], [1, 1], "same"), True))
    x3 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x3, 4*depth, [3, 3], [1, 1], "same"), True))
    x4 = eddl.MaxPool(x3, [2, 2], [2, 2])#eddl.MaxPool(x3, [2, 2], [2, 2])#
    x4 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x4, 8*depth, [3, 3], [1, 1], "same"), True))
    x4 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x4, 8*depth, [3, 3], [1, 1], "same"), True))
    x5 = eddl.MaxPool(x4, [2, 2], [2, 2])#eddl.MaxPool(x4, [2, 2], [2, 2])#

    # middle conv
    x5 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x5, 16*depth, [3, 3], [1, 1], "same"), True))
    x5 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x5, 16*depth, [3, 3], [1, 1], "same"), True))

    # decoder
    x5 = eddl.Conv(
        eddl.UpSampling(x5, [2, 2]), 8*depth, [2, 2], [1, 1], "same"
    )
    x4 = eddl.Concat([x4, x5])
    x4 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x4, 8*depth, [3, 3], [1, 1], "same"), True))
    x4 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x4, 8*depth, [3, 3], [1, 1], "same"), True))
    x4 = eddl.Conv(
        eddl.UpSampling(x4, [2, 2]), 4*depth, [2, 2], [1, 1], "same"
    )
    x3 = eddl.Concat([x3, x4])
    x3 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x3, 4*depth, [3, 3], [1, 1], "same"), True))
    x3 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x3, 4*depth, [3, 3], [1, 1], "same"), True))
    x3 = eddl.Conv(
        eddl.UpSampling(x3, [2, 2]), 2*depth, [2, 2], [1, 1], "same"
    )
    x2 = eddl.Concat([x2, x3])
    x2 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x2, 2*depth, [3, 3], [1, 1], "same"), True))
    x2 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x2, 2*depth, [3, 3], [1, 1], "same"), True))
    x2 = eddl.Conv(
        eddl.UpSampling(x2, [2, 2]), depth, [2, 2], [1, 1], "same"
    )
    x = eddl.Concat([x, x2])
    x = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x, depth, [3, 3], [1, 1], "same"), True))
    x = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x, depth, [3, 3], [1, 1], "same"), True))

     # final conv
    x = eddl.Sigmoid(eddl.Conv(x, num_classes, [1, 1]))
