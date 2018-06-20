# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import json
import logging
import os

import chainer
import click
import dill  # This is for joblib to use dill. Do NOT delete it.
import numpy as np
from chainer import training
from chainer.training import extensions
from joblib import Memory

import rationale
from rationale.dataset import prepare_data
from rationale.training import SaveRestore, EarlyStoppingTrigger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.argument('aspect', type=int)
@click.argument('word2vec', type=click.Path(exists=True))
@click.argument('trained-model', type=click.Path(exists=True))
@click.argument('test', type=click.Path(exists=True))
@click.option('--out', '-o', default='result',
              help='Directory to output the result and temporaly file')
@click.option('--gpu', '-g', type=int, default=-1,
              help='GPU ID (negative value indicates CPU)')
@click.option('--batchsize', '-b', type=int, default=256,
              help='Number of images in each mini-batch')
@click.option('--sparsity-coef', type=float, default=0.0003,
              help='Sparsity cost coefficient lambda_1')
@click.option('--coherent-coef', type=float, default=2.0,
              help='Coherence cost coefficient lambda_2')
@click.option('--order', type=int, default=2,
              help='Order of RCNN')
def run(aspect, word2vec, trained_model, gpu, out, test, batchsize,
        sparsity_coef, coherent_coef, order):
    """
    Train "Rationalizing Neural Predictions" for one specified aspect.

    Please refer README.md for details.
    """
    memory = Memory(cachedir='.', verbose=1)
    w2v, vocab, test_dataset, _, _ = \
        memory.cache(prepare_data)(test, word2vec, aspect)

    encoder = rationale.models.Encoder(
        w2v.shape[1], order, 200, 2, dropout=0.1
    )
    # Original impl. uses two layers to model bi-directional LSTM
    generator = rationale.models.Generator(
        w2v.shape[1], order, 200, dropout=0.1
    )
    model = rationale.models.RationalizedRegressor(
        generator, encoder, w2v.shape[0], w2v.shape[1], initialEmb=w2v,
        dropout_emb=0.1,
        sparsity_coef=sparsity_coef, coherent_coef=coherent_coef
    )
    # Resume from a snapshot
    chainer.serializers.load_npz(trained_model, model)

    if gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    inv_vocab = {v: k for k, v in vocab.items()}
    results = rationale.prediction.test(model, test_dataset, inv_vocab,
                                        device=gpu, batchsize=batchsize)
    with open(os.path.join(out, 'output.json'), 'w') as fout:
        json.dump(results, fout)


if __name__ == '__main__':
    run()
