#! /usr/bin/env python
# -*- conding:utf-8 -*-

import math
import chainer


class CosineShift(chainer.training.extension.Extension):
    def __init__(self, attr, period, period_mult=1, optimizer=None):
        self._attr = attr
        self._period = period
        self._period_mult = period_mult
        self._optimizer = optimizer
        self._init_attr = None

    def initialize(self, trainer):
        self._update_value(trainer)

    def __call__(self, trainer):
        self._update_value(trainer)

    def serialize(self, serializer):
        self._period = serializer("_period", self._period)
        self._period_mult = serializer("_period_multi", self._period_mult)

    def _update_value(self, trainer):
        optimizer = self._optimizer or trainer.updater.get_optimizer('main')
        if self._init_attr is None:
            self._init_attr = getattr(optimizer, self._attr)
        epoch = trainer.updater.epoch

        period_range = self._period
        period_start = 0
        period_end = period_range

        while period_end <= epoch:
            period_start = period_end
            period_range *= self._period_mult
            period_end += period_range

        t_cur = epoch - period_start
        t_i = period_range
        value = self._init_attr * \
            (0.5 + 0.5 * math.cos((t_cur / t_i) * math.pi))

        setattr(optimizer, self._attr, value)
