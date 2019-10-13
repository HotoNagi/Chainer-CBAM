#! /usr/bin/env python
# -*- coding:utf-8 -*-

#-------------------------------------------------------#
#                        Import                         #
#-------------------------------------------------------#
import chainer
import chainer.functions as F
from chainer import link
import chainer.links as L
from chainer.serializers import npz
import chainer.links.model.vision.resnet as R
from chainer import initializers
from chainer.functions.array.reshape import reshape
from chainer import Sequential
from chainer import reporter

from PIL import Image
import random
import numpy as np
import cupy as cp
import math
import cv2
from chainer import Variable
#-------------------------------------------------------#


#-------------------------------------------------------#
#                    ResNet50_CBAM                      #
#-------------------------------------------------------#

class ResNet50_CBAM(chainer.Chain):
    def __init__(self, pretrained_model='../../.chainer/dataset/pfnet/chainer/models/ResNet-50-model.npz',
                 output=8, blockexpansion = 4):
        super(ResNet50_CBAM, self).__init__()
        with self.init_scope():

            self.base = BaseResNet50()

            self.conv1 = self.base.conv1
            self.bn1 = self.base.bn1
            self.layer1 = self.base.res2
            self.layer2 = self.base.res3
            self.layer3 = self.base.res4
            self.layer4 = self.base.res5
            self.fc = L.Linear(512*blockexpansion, output)
            npz.load_npz(file=pretrained_model, obj=self.base, strict=False)

    def forward(self, x):

        h = self.bn1(self.conv1(x))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2, pad=1)

        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        self.cam = h
        h = _global_average_pooling_2d(h)
        h = self.fc(h)

        return h


class BaseResNet50(chainer.Chain):

    def __init__(self):
        super(BaseResNet50, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 64, 7, 2, 3)
            self.bn1 = L.BatchNormalization(64)
            self.res2 = BuildingBlock(3, 64, 64, 256, 1)
            self.res3 = BuildingBlock(4, 256, 128, 512, 2)
            self.res4 = BuildingBlock(6, 512, 256, 1024, 2)
            self.res5 = BuildingBlock(3, 1024, 512, 2048, 2)

    def forward(self, x):

            h = self.bn1(self.conv1(x))
            h = F.max_pooling_2d(F.relu(h), 3, stride=2)
            h = self.res2(h)
            h = self.res3(h)
            h = self.res4(h)
            h = self.res5(h)

            return h


class BuildingBlock(link.Chain):

    def __init__(self, n_layers=None, in_channels=None, mid_channels=None,
                 out_channels=None, stride=None, initialW=None,
                 downsample_fb=None, use_cbam=True, **kwargs):
        super(BuildingBlock, self).__init__()

        if 'n_layer' in kwargs:
            warnings.warn(
                'Argument `n_layer` is deprecated. '
                'Please use `n_layers` instead',
                DeprecationWarning)
            n_layers = kwargs['n_layer']

        with self.init_scope():
            self.a = BottleneckA(
                in_channels, mid_channels, out_channels, stride,
                initialW, downsample_fb)
            self._forward = ['a']
            for i in range(n_layers - 1):
                name = 'b{}'.format(i + 1)
                bottleneck = BottleneckB(out_channels, mid_channels, initialW)
                setattr(self, name, bottleneck)
                self._forward.append(name)

    def forward(self, x):
        for name in self._forward:
            l = getattr(self, name)
            x = l(x)
        return x


class BottleneckA(link.Chain):

    def __init__(self, in_channels, mid_channels, out_channels,
                 stride=2, initialW=None, downsample_fb=False):
        super(BottleneckA, self).__init__()

        stride_1x1, stride_3x3 = (1, stride) if downsample_fb else (stride, 1)
        w = initializers.HeNormal()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_channels, mid_channels, 1, stride_1x1, 0, initialW=initialW,
                nobias=True)
            self.bn1 = L.BatchNormalization(mid_channels)
            self.conv2 = L.Convolution2D(
                mid_channels, mid_channels, 3, stride_3x3, 1,
                initialW=initialW, nobias=True)
            self.bn2 = L.BatchNormalization(mid_channels)
            self.conv3 = L.Convolution2D(
                mid_channels, out_channels, 1, 1, 0, initialW=initialW,
                nobias=True)
            self.bn3 = L.BatchNormalization(out_channels)
            self.conv4 = L.Convolution2D(
                in_channels, out_channels, 1, stride, 0, initialW=initialW,
                nobias=True)
            self.bn4 = L.BatchNormalization(out_channels)

    def forward(self, x):
        h1 = F.relu(self.bn1(self.conv1(x)))
        h1 = F.relu(self.bn2(self.conv2(h1)))
        h1 = self.bn3(self.conv3(h1))
        h2 = self.bn4(self.conv4(x))
        return F.relu(h1 + h2)


class BottleneckB(link.Chain):

    def __init__(self, in_channels, mid_channels, initialW=None):
        super(BottleneckB, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_channels, mid_channels, 1, 1, 0, initialW=initialW,
                nobias=True)
            self.bn1 = L.BatchNormalization(mid_channels)
            self.conv2 = L.Convolution2D(
                mid_channels, mid_channels, 3, 1, 1, initialW=initialW,
                nobias=True)
            self.bn2 = L.BatchNormalization(mid_channels)
            self.conv3 = L.Convolution2D(
                mid_channels, in_channels, 1, 1, 0, initialW=initialW,
                nobias=True)
            self.bn3 = L.BatchNormalization(in_channels)
            self.ca = ChannelAttention(in_channels)
            self.sa = SpatialAttention()

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))
        
        h = self.ca(h) * h
        h = self.sa(h) * h
        
        return F.relu(h + x)

#-------------------------------------------------------#

#-------------------------------------------------------#
#               Channel Attention Module                #
#-------------------------------------------------------#

class ChannelAttention(chainer.Chain):
    def __init__(self, in_planes, reduction_ratio=16):
        w = chainer.initializers.Normal(scale=0.01)
        super(ChannelAttention, self).__init__()
        with self.init_scope():
            self.fc1 = L.Convolution2D(
                in_planes, in_planes // reduction_ratio, ksize=1,initialW=w,
                nobias=True)
            self.fc2 = L.Convolution2D(
                in_planes // reduction_ratio, in_planes, ksize=1,initialW=w,
                nobias=True)
        """
            self.mlp = Sequential(
                F.flatten,
                L.Linear(in_planes, in_planes // reduction_ratio),
                F.relu,
                L.Linear(in_planes // reduction_ratio, in_planes)
            )
        """

    def forward(self, x):
        avg_pool = AdaptiveAvgPool2d(x)
        avg_out = self.fc2(F.relu(self.fc1(avg_pool)))
        max_pool = AdaptiveMaxPool2d(x)
        max_out = self.fc2(F.relu(self.fc1(max_pool)))
        out = avg_out + max_out
        return F.sigmoid(out)

#-------------------------------------------------------#


#-------------------------------------------------------#
#               Spatial Attention Module                #
#-------------------------------------------------------#

class SpatialAttention(chainer.Chain):
    def __init__(self, kernel_size=7):
        w = chainer.initializers.Normal(scale=0.01)
        super(SpatialAttention, self).__init__()
        
        with self.init_scope():
            assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
            padding = 3 if kernel_size == 7 else 1
            self.conv1 = L.Convolution2D(
                2, 1, ksize=kernel_size, pad=padding, initialW=w,
                nobias=True)
    def forward(self, x):
        avg_out = F.mean(x, axis=1, keepdims=True)
        max_out = F.max(x, axis=1, keepdims=True)
        x = F.concat([avg_out, max_out], axis=1)
        x = self.conv1(x)
        return F.sigmoid(x)

#-------------------------------------------------------#

#-------------------------------------------------------#
#                       Pooling                         #
#-------------------------------------------------------#

def AdaptiveAvgPool2d(x):
    n, channel, rows, cols = x.shape
    h = F.average_pooling_2d(x, (rows, cols), (rows, cols))
    return h

def AdaptiveMaxPool2d(x):
    n, channel, rows, cols = x.shape
    h = F.max_pooling_2d(x, (rows, cols), (rows, cols))
    return h


def _global_average_pooling_2d(x):
    n, channel, rows, cols = x.shape
    h = F.average_pooling_2d(x, (rows, cols), stride=1)
    h = reshape(h, (n, channel))
    return h

#-------------------------------------------------------#
