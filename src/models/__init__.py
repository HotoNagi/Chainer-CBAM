#! /usr/bin/env python
# -*- conding:utf-8 -*-

import sys
import os

from .resnet50_cbam import ResNet50_CBAM

sys.path.append(os.pardir)

archs = {
         'resnet50_cbam': ResNet50_CBAM
        }
