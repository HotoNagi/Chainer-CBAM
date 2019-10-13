#! /usr/bin/env python
# -*- conding:utf-8 -*-
import os
import sys
import chainer
import numpy as np

import cv2
import math
import random

from models import archs

from functools import partial
from chainercv import transforms


def get_dataset(train_data, test_data, root, datasets, use_mean=True):

    mean_path = root + '/mean.npy'
    if os.path.exists(mean_path):
        mean = np.load(mean_path)
    else:
        mean = compute_mean(datasets, root)
        np.save(mean_path, mean)
    print('use mean flag is ', use_mean)
    if not use_mean:
        print('not using mean')

    train = chainer.datasets.TransformDataset(
        train_data, partial(_transform,
                            mean=mean, train=True, mean_flag=use_mean))
    test = chainer.datasets.TransformDataset(
        test_data, partial(_transform,
                           mean=mean, train=False, mean_flag=use_mean))

    return train, test, mean


# 画像平均計算
def compute_mean(datasets, root, size=(224, 224)):

    print('画像平均計算...')
    sum_image = 0
    N = len(datasets)
    for i, (image, _) in enumerate(datasets):
        # imgのリサイズ
        image = image.transpose(1, 2, 0)
        image = resize(image, size)
        sum_image += image
        sys.stderr.write('{} / {}\r'.format(i+1, N))
        sys.stderr.flush()
    sys.stderr.write('\n')
    mean = sum_image / N


    return mean


# 前処理
def _transform(data, mean, train=True, mean_flag=False):
    
    img, label = data
    img = img.copy()

    size316 = (316, 316)
    size = (224, 224)

    img = transforms.scale(img, 316)
    img = transforms.center_crop(img, size316)

    # 学習のときだけ実行
    if train:
        img = transforms.random_rotate(img)
        img = transforms.random_flip(img, x_random=True, y_random=True)
    # imgのリサイズ
    img = img.transpose(1, 2, 0)
    img = resize(img, size)

    # 画像から平均を引く
    if mean_flag:
        img -= mean
        print("mean use")

    img *= (1.0 / 255.0)

    img = img.transpose(2, 0, 1)

    return img, label


# 回転の角度範囲
angle_range = [i for i in range(0, 360, 10)]


# ランダム回転
def random_rotate(img):

    img = img.copy()
    h, w, _ = img.shape
    angle = np.random.choice(angle_range)

    center = (h / 2, w / 2)

    # 回転変換行列の算出
    # cv2.getRotationMatrix2D(画像の中心の座標,回転させたい角度,拡大比率)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # アフィン変換
    img = cv2.warpAffine(img, rotation_matrix, img.shape[:2])

    rad = angle * np.pi / 180
    new_length = int(h / (np.abs(np.cos(rad)) + np.abs(np.sin(rad))))

    half_new_length = new_length // 2
    img = img[w // 2 - half_new_length: w // 2 + half_new_length,
              h // 2 - half_new_length: h // 2 + half_new_length]

    return img


def random_flip(img, y_random=False, x_random=False, copy=False):

    y_flip, x_flip = False, False
    if y_random:
        y_flip = np.random.choice([True, False])
    if x_random:
        x_flip = np.random.choice([True, False])

    if y_flip:
        img = img[::-1, :, :]
    if x_flip:
        img = img[:, ::-1, :]

    if copy:
        img = img.copy()

    return img


def random_erase(img, p=0.5, s_l=0.02, s_h=0.4, r1=0.3, r2=1. / 0.3):

    img = img.copy()
    p1 = np.random.uniform(0, 1)
    if p1 < p:
        return img

    else:
        while True:
            h, w, _ = img.shape
            S_e = random.uniform(s_l, s_h) * (h * w)
            r_e = random.uniform(r1, r2)
            H_e = int(math.sqrt(S_e * r_e))
            W_e = int(math.sqrt(S_e / r_e))

            x_e = random.randint(0, w)
            y_e = random.randint(0, h)

            if ((x_e + W_e) <= w) and (y_e + H_e) <= h:
                top = x_e
                bottom = x_e + W_e
                left = y_e
                right = y_e + H_e

                img[top:bottom, left:right, :] = np.random.uniform(0, 1)

                return img


def resize(img, size):

    img = cv2.resize(img, size)
    return img


# cutmix
def get_dataset_cutmix(train_data, test_data, root, datasets, class_num, cutmix_alpha, use_mean=True):

    mean_path = root + '/mean.npy'
    if os.path.exists(mean_path):
        mean = np.load(mean_path)
    else:
        mean = compute_mean(datasets, root)
        np.save(mean_path, mean)
    print('use mean flag is ', use_mean)
    if not use_mean:
        print('not using mean')

    train = CutMixDataset(pairs=train_data, class_num=class_num, cutmix_alpha=cutmix_alpha, mean=mean,
                          mean_flag=use_mean)
    test = chainer.datasets.TransformDataset(
        test_data, partial(_transform,
                           mean=mean, train=False, mean_flag=use_mean))

    return train, test, mean


class CutMixDataset(chainer.dataset.DatasetMixin):

    def __init__(self, pairs, class_num, cutmix_alpha, mean, mean_flag):
        self.base = pairs
        self.class_num = class_num
        self.cutmix_alpha = cutmix_alpha
        self.mean = mean
        self.mean_flag= mean_flag
        

    def __len__(self):
        return len(self.base)
    
    def get_example(self, i):
        cutmix_alpha = self.cutmix_alpha
        class_num = self.class_num
        h = 224
        w = 224
        img_1, label_1 = self.base[i]
        while True:
            img_2, label_2 = random.choice(self.base)
            if label_1 != label_2:
                break

        img_1 = transforms.scale(img_1, 316)
        img_2 = transforms.scale(img_2, 316)

        img_1 = transforms.center_crop(img_1, (316, 316))
        img_2 = transforms.center_crop(img_2, (316, 316))

        img_1 = transforms.random_flip(img_1, x_random=True, y_random=True)
        img_2 = transforms.random_flip(img_2, x_random=True, y_random=True)

        img_1 = img_1.transpose(1, 2, 0)
        img_2 = img_2.transpose(1, 2, 0)

        #img_1 = random_rotate(img_1)
        #img_2 = random_rotate(img_2)

        img_1 = img_1.transpose(2, 0, 1)
        img_2 = img_2.transpose(2, 0, 1)

        img_1 = transforms.resize(img_1, (224, 224))
        img_2 = transforms.resize(img_2, (224, 224))

        img_1 = transforms.random_rotate(img_1)
        img_2 = transforms.random_rotate(img_2)

        #####################################################
        # cutmix実装
        #####################################################

        # sample the bounding box coordinates
        while True:
            # the combination ratio λ between two data points is 
            # sampled from the beta distribution Beta(α, α)
            l = np.random.beta(cutmix_alpha, cutmix_alpha)
            rx = random.randint(0, w)
            rw = w * math.sqrt(1-l)
            ry = random.randint(0, h)
            rh = h * math.sqrt(1-l)
            if ((ry + round(rh)) < h)and((rx + round(rw)) < w):
                break

        # denotes a binary mask indicating where to drop out
        # and fill in from two images
        M_1 = np.zeros((h, w))
        M_1[ry:(ry + round(rh)),rx:(rx + round(rw))] = 1

        M = np.ones((h, w))
        M = M - M_1

        # Define the combining operation.
        img = (img_1 * M + img_2 * M_1).astype(np.float32)

        # eye関数は単位行列を生成する
        eye = np.eye(class_num)
        label = (eye[label_1] * l + eye[label_2] * (1 - l)).astype(np.float32)
        # 画像255で割る（輝度値をすべて0～1の間に収める）
        img = img.transpose(1, 2, 0)
        # 画像から平均を引く
        if self.mean_flag:
            img -= self.mean

        img = img / 255.0
        img = img.transpose(2, 0, 1)

        return img, label
