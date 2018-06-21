# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import json
import logging

import chainer
import click
import dill  # This is for joblib to use dill. Do NOT delete it.
from joblib import Memory

import rationale
from rationale.dataset import prepare_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.argument('aspect', type=int)
@click.argument('word2vec', type=click.Path(exists=True))
@click.argument('trained-model', type=click.Path(exists=True))
@click.argument('test', type=click.Path(exists=True))
@click.option('--out', '-o', default='result/statistics.json')
@click.option('--gpu', '-g', type=int, default=-1,
              help='GPU ID (negative value indicates CPU)')
@click.option('--batchsize', '-b', type=int, default=256,
              help='Number of images in each mini-batch')
@click.option('--sparsity-coef', type=float, default=0.0003,
              help='Sparsity cost coefficient lambda_1')
@click.option('--coherent-coef', type=float, default=2.0,
              help='Coherence cost coefficient lambda_2')
@click.option('--dependent', is_flag=True,
              help='Use a variation of the generator that are dependent on'
                   'previously sampled tokens')
@click.option('--order', type=int, default=2,
              help='Order of RCNN')
def run(aspect, word2vec, trained_model, gpu, test, out, batchsize,
        sparsity_coef, coherent_coef, dependent, order):
    """
    Train "Rationalizing Neural Predictions" for one specified aspect.

    Please refer README.md for details.
    """
    memory = Memory(cachedir='.', verbose=1)
    w2v, vocab, _, _, test_dataset = \
        memory.cache(prepare_data)(None, word2vec, aspect, annotation=test)

    encoder = rationale.models.Encoder(
        w2v.shape[1], order, 200, 2, dropout=0.1
    )
    generator_cls = (rationale.models.GeneratorDependent
                     if dependent else rationale.models.Generator)
    # Original impl. uses two layers to model bi-directional LSTM
    generator = generator_cls(w2v.shape[1], order, 200, dropout=0.1)
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

    result = rationale.prediction.evaluate_rationale(
        model, test_dataset, device=gpu, batchsize=batchsize)
    print(json.dumps(result, indent=2))
    with open(out, 'w') as fout:
        json.dump(result, fout, indent=2)


if __name__ == '__main__':
    run()
