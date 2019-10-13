#! /usr/bin/env python
# -*- conding:utf-8 -*-

import chainer
from chainer.dataset import concat_examples
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import collections


def confusion_matrix_cocoa(validation, gpu_id, batchsize, model, save_path, nfold, labels_list):
    
    val_results = {'y_pred': [], 'y_true': []}
    # testデータをSerialIteratorに渡す
    test_iter = chainer.iterators.SerialIterator(
        validation, batchsize, repeat=False, shuffle=False)

    while True:
        X_test_batch = test_iter.next()
        X_test, y_test = concat_examples(X_test_batch, gpu_id)

        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            y_pred = model.predictor(X_test)
            # y_pred = model.predictor(X_test[None, ...]).data.argmax(axis=1)[0] この書き方はダメ？

        y_pred = chainer.cuda.to_cpu(y_pred.data)

        val_results['y_pred'].extend(np.argmax(y_pred, axis=1).tolist())
        val_results['y_true'].extend(y_test.tolist())

        if test_iter.is_new_epoch:
            test_iter.reset()
            break

    labels = sorted(list(set(val_results['y_true'])))

    matrix = confusion_matrix(
        val_results['y_true'], val_results['y_pred'], labels=labels)
    print("accuracy score")
    print(accuracy_score(val_results['y_true'], val_results['y_pred']))
    matrix = matrix.astype(np.float64)
    for i in range(len(labels)):
        num_of_true = 0
        for j in range(len(labels)):
            num_of_true += matrix[i][j]
        for j in range(len(labels)):
            matrix[i][j] = matrix[i][j] / num_of_true

    num_of_true = collections.Counter(val_results['y_true'])
    df_cmx = pd.DataFrame(matrix, index=labels_list, columns=labels_list)

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 20
    plt.figure(figsize=(24, 18))
    ax = sn.heatmap(df_cmx, annot=True, fmt='.3f', cmap='jet', linewidth='.5')

    ax.set_ylim(len(labels_list), 0)
    ax.set_xticklabels(labels_list, rotation=30)
    ax.set_yticklabels(labels_list, rotation=0)
    plt.tight_layout()
    plt.savefig(save_path / "{}-fold_heatmap.png".format(nfold + 1))
    save_matrix = save_path / "{}-fold_matrix.npy".format(nfold + 1)
    np.save(save_matrix, matrix)