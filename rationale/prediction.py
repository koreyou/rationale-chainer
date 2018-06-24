# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import logging

import chainer
import numpy as np
from chainer.configuration import using_config
from chainer.cuda import to_cpu
from chainer.iterators import SerialIterator

from rationale.training import convert

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test(model, dataset, inv_vocab, device=-1, batchsize=128):
    """
    Predict without evaluating. Refer :func:`test` for the information about
    arguments.

    Returns:
        numpy.ndarray: Prediction probability whose size is `data size` x
            `number of labels`.

    """
    if device >= 0:
        model.to_gpu(device)

    it = SerialIterator(dataset, batchsize, repeat=False, shuffle=False)

    results = []
    for batch in it:
        in_arrays = convert(batch, device)
        with chainer.function.no_backprop_mode(), using_config('train', False):
            y, z_prob, z = model.forward(in_arrays['xs'])
            loss, loss_encoder, sparsity, coherence, regressor_cost, loss_generator = \
                model.calc_loss(y, z, z_prob, in_arrays['ys'])
        loss = to_cpu(loss.data)
        loss_encoder = to_cpu(loss_encoder.data)
        sparsity = to_cpu(sparsity)
        coherence = to_cpu(coherence)
        regressor_cost = to_cpu(regressor_cost)
        loss_generator = to_cpu(loss_generator.data)
        y = to_cpu(y.data).tolist()
        z = [to_cpu(zi).tolist() for zi in z]
        xs = [to_cpu(xi).tolist() for xi in in_arrays['xs']]

        results.extend(({
            'x': xs[i],
            'z': list(map(int, z[i])),
            'y': y[i],
            'text': [inv_vocab[t] for t in xs[i]],
            'rationale': [inv_vocab[t] if zt > 0.5 else '_'
                          for t, zt in zip(xs[i], z[i])],
            'loss': float(loss[i]),
            'loss_encoder': float(loss_encoder[i]),
            'sparsity_cost': float(sparsity[i]),
            'coherence': float(coherence[i]),
            'regressor_cost': float(regressor_cost[i]),
            'loss_generator': float(loss_generator[i])
        } for i in range(len(y))))
    return results


def evaluate_rationale(model, dataset, device=-1, batchsize=128):
    if device >= 0:
        model.to_gpu(device)
    it = SerialIterator(dataset, batchsize, repeat=False, shuffle=False)

    tot_mse = 0.0
    accum_precision = 0.0  # for calculating macro precision
    true_positives = 0.0  # for calculating micro precision
    chosen_ratios = 0.0  # for calculating micro precision
    tot_z, tot_n, tot_t = 1e-10, 1e-10, 1e-10
    for batch in it:
        in_arrays = convert(batch, device)
        with chainer.function.no_backprop_mode(), using_config('train', False):
            pred, z_prob, z = model.forward(in_arrays['xs'])
            regressor_cost = model.calc_loss(pred, z, z_prob, in_arrays['ys'])[4]
        regressor_cost = to_cpu(regressor_cost)
        z = [to_cpu(zi).tolist() for zi in z]

        tot_mse += regressor_cost.sum()

        for bi, zi in zip(batch, z):
            true_z = bi['zs']
            nzi = sum(zi)
            tp = np.logical_and(zi, true_z)
            if nzi == 0:
                # precision is undefined when there is 0 prediction
                continue
            accum_precision += tp / float(nzi)
            tot_n += 1
            true_positives += tp
            tot_z += nzi
            chosen_ratios += nzi / float(len(zi))
            tot_t += len(zi)

    result = {
        "mse": tot_mse/len(dataset),
        "macro_precision": accum_precision / tot_n,
        "micro_precision": true_positives / tot_z,
        "micro_chosen_ratio": tot_z / tot_t,
        "macro_chosen_ratio": chosen_ratios / tot_n,
    }
    return result
