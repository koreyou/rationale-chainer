# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import logging

import chainer
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
    #import pdb; pdb.set_trace()
    if device >= 0:
        model.to_gpu(device)

    it = SerialIterator(dataset, batchsize, repeat=False, shuffle=False)

    results = []
    for batch in it:
        in_arrays = convert(batch, device)
        with chainer.function.no_backprop_mode(), using_config('train', False):
            y, z = model.forward(in_arrays['xs'])
            loss, loss_encoder, sparsity, coherence, regressor_cost, loss_generator = \
                model.calc_loss(y, z, in_arrays['ys'])
        loss = to_cpu(loss.data)
        loss_encoder = to_cpu(loss_encoder.data)
        sparsity = to_cpu(sparsity)
        coherence = to_cpu(coherence)
        regressor_cost = to_cpu(regressor_cost)
        loss_generator = to_cpu(loss_generator.data)
        y = to_cpu(y.data).tolist()
        z = [to_cpu(zi.data).tolist() for zi in z]
        xs = [to_cpu(xi).tolist() for xi in in_arrays['xs']]

        results.extend(({
            'x': xs[i],
            'z': z[i],
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
