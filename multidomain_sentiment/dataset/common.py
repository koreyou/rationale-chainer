# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import numpy as np
from chainer.datasets import DictDataset


def create_dataset(texts, labels, domains, size=-1):

    if size > 0:
        # Sample data AFTER all data has been loaded. This is because
        # There might be bias in data ordering.
        ind = np.random.permutation(len(texts))[:size]
        if labels is None:
            return DictDataset(
                xs=[texts[i] for i in ind],
                domains=[domains[i] for i in ind])
        else:
            return DictDataset(
                xs=[texts[i] for i in ind], ys=[labels[i] for i in ind],
                domains=[domains[i] for i in ind])
    else:
        if labels is None:
            return DictDataset(xs=texts, domains=domains)
        else:
            return DictDataset(xs=texts, ys=labels, domains=domains)
