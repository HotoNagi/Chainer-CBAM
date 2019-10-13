#! /usr/bin/env python
# -*- conding:utf-8 -*-

import matplotlib

import os
import glob
import shutil
import datetime
import numpy as np
from pathlib import Path
from utils.get_dataset import get_dataset

import chainer
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.datasets import LabeledImageDataset
from chainercv.datasets import DirectoryParsingLabelDataset
from sklearn.model_selection import StratifiedKFold

matplotlib.use('Agg')

from models import archs
from utils.args import parser
from utils.cosine_shift import CosineShift
from utils.confusion_matrix_cocoa import confusion_matrix_cocoa

def main():

    args = parser()
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    save_dir = Path('result') / now
    log_dir = save_dir / 'log'
    model_dir = save_dir / 'model'
    snap_dir = save_dir / 'snap'
    matrix_dir = save_dir / 'matrix'

    save_dir.mkdir(exist_ok=True, parents=True)
    log_dir.mkdir(exist_ok=True, parents=True)
    model_dir.mkdir(exist_ok=True, parents=True)
    snap_dir.mkdir(exist_ok=True, parents=True)
    matrix_dir.mkdir(exist_ok=True, parents=True)


    root = args.dataset

    dir_list = os.listdir(root)
    dir_list.sort()

    if 'mean.npy' in dir_list:
        dir_list.remove('mean.npy')

    print('dataset loading ...')
    datasets = DirectoryParsingLabelDataset(root)
    print('finish!')

    class_num = len(set(datasets.labels))
    print('class number : {}'.format(class_num))


    k_fold = args.kfold
    print('k_fold : {}'.format(k_fold))

    X = np.array([image_paths for image_paths in datasets.img_paths])
    y = np.array([label for label in datasets.labels])

    kfold = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=402).split(X, y)
    for k, (train_idx, val_idx) in enumerate(kfold):

        print("============= {} fold training =============".format(k + 1))
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        train = LabeledImageDataset([(x, y) for x, y in zip(X_train, y_train)])
        validation = LabeledImageDataset([(x, y) for x, y in zip(X_val, y_val)])

        train, validation, mean = get_dataset(train, validation, root, datasets, use_mean=False)


        model = L.Classifier(archs[args.arch](output=class_num))
        lr = args.lr
        optimizer = chainer.optimizers.MomentumSGD(lr)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0001))

        if args.gpu >= 0:
            chainer.cuda.get_device_from_id(args.gpu).use()
            model.to_gpu()

        train_iter = chainer.iterators.MultithreadIterator(train, args.batchsize, n_threads=8)
        validation_iter = chainer.iterators.MultithreadIterator(validation, args.batchsize,
                                                                repeat=False, shuffle=False, n_threads=8)
        updater = training.StandardUpdater(
            train_iter, optimizer, device=args.gpu)
        trainer = training.Trainer(
            updater, (args.epoch, 'epoch'), out=save_dir)

        log_trigger = (1, 'epoch')
        target = 'lr'
        trainer.extend(CosineShift(target, args.epoch, 1),
                       trigger=(1, "epoch"))

        trainer.extend(extensions.Evaluator(validation_iter, model, device=args.gpu),
                       trigger=log_trigger)

        snap_name = '{}-{}_fold_model.npz'.format(k_fold, k+1)
        trainer.extend(extensions.snapshot_object(model, str(snap_name)),
                       trigger=chainer.training.triggers.MaxValueTrigger(
                       key='validation/main/accuracy', trigger=(1, 'epoch')))

        log_name = '{}-{}_fold_log.json'.format(k_fold, k+1)
        trainer.extend(extensions.LogReport(
            log_name=str(log_name), trigger=log_trigger))

        trainer.extend(extensions.observe_lr(), trigger=log_trigger)
        trainer.extend(extensions.PrintReport([
            'epoch', 'iteration',
            'main/loss', 'validation/main/loss',
            'main/accuracy', 'validation/main/accuracy',
            'elapsed_time', 'lr'
        ]), trigger=(1, 'epoch'))

        trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                             'epoch',file_name='loss{}.png'.format(k+1)))
        trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'],
                                             'epoch', file_name='accuracy{}.png'.format(k+1)))
        trainer.extend(extensions.ProgressBar(update_interval=10))
 
        trainer.run()


        snap_file = save_dir / snap_name
        shutil.move(str(snap_file), str(snap_dir))

        log_file = save_dir / log_name
        shutil.move(str(log_file), str(log_dir))

        save_model = model_dir / "{}_{}-{}_fold.npz".format(now, k_fold, k + 1)
        chainer.serializers.save_npz(str(save_model), model)

        print("============= {} fold Evaluation =============".format(k + 1))

        dnames = glob.glob('{}/*'.format(root))
        labels_list = []
        for d in dnames:
            p_dir = Path(d)
            labels_list.append(p_dir.name)

        if 'mean.npy' in labels_list:
            labels_list.remove('mean.npy')

        confusion_matrix_cocoa(validation, args.gpu, 7,
                               model, matrix_dir, k, labels_list)


if __name__ == '__main__':
    main()
