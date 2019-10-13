#! /usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
from models import archs

def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset',
                        default='../../good_condition',
                        help="Choise input dataset by full path")

    parser.add_argument('--kfold', '-k',
                        type=int, default=4,
                        help="Number of k fold")

    parser.add_argument('--gpu', '-g',
                        type=int, default=-0,
                        help="Number of using gpu_id")

    parser.add_argument('--arch', '-a',
                        choices=archs.keys(), default='resnet50_cbam',
                        help="Name of using training model")

    parser.add_argument('--lr',
                        type=float, default=5e-05,
                        help="Learning rate parameter")

    parser.add_argument('--batchsize', '-b',
                        type=int, default=32,
                        help="Number of using batchsize")

    parser.add_argument('--epoch', '-e',
                        type=int, default=200,
                        help="Number of using epoch")

    parser.add_argument('--resume', '-r',
                        default='',
                        help="")

    args = parser.parse_args()

    return args
