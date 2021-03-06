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
import chainer.backends.intel64
from joblib import Memory

import rationale
from rationale.dataset import prepare_data
from rationale.training import SaveRestore, ConditionalRestart
from rationale.minmax_value_trigger import MinValueTrigger


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.argument('aspect', type=int)
@click.argument('train', type=click.Path(exists=True))
@click.argument('word2vec', type=click.Path(exists=True))
@click.option('--epoch', '-e', type=int, default=100,
              help='Number of sweeps over the dataset to train')
@click.option('--frequency', '-f', default=[1, 'epoch'],
              type=(int, click.Choice(['epoch', 'iteration'])),
              help='Frequency of taking a snapshot')
@click.option('--gpu', '-g', type=int, default=-1,
              help='GPU ID (negative value indicates CPU)')
@click.option('--out', '-o', default='result',
              help='Directory to output the result and temporaly file')
@click.option('--batchsize', '-b', type=int, default=256,
              help='Number of images in each mini-batch')
@click.option('--lr', type=float, default=0.005,
              help='Unnormalized learning rate to batch size')
@click.option('--sparsity-coef', type=float, default=0.0003,
              help='Sparsity cost coefficient lambda_1')
@click.option('--coherent-coef', type=float, default=2.0,
              help='Coherence cost coefficient lambda_2 against sparsity cost '
                   'coefficient, i.e. lambda_2 / lambda_1')
@click.option('--fix_embedding', type=bool, default=False,
              help='Fix word embedding during training')
@click.option('--dependent', is_flag=True,
              help='Use a variation of the generator that are dependent on'
                   'previously sampled tokens')
@click.option('--order', type=int, default=2,
              help='Order of RCNN')
@click.option('--resume', '-r', default='',
              help='Resume the training from snapshot')
def run(aspect, train, word2vec, epoch, frequency, gpu, out, batchsize,
        lr, sparsity_coef, coherent_coef, fix_embedding, dependent, order,
        resume):
    """
    Train "Rationalizing Neural Predictions" for one specified aspect.

    Please refer README.md for details.
    """
    memory = Memory(cachedir='.', verbose=1)
    w2v, vocab, dataset, _, _ = \
        memory.cache(prepare_data)(train, word2vec, aspect)
    train_dataset, dev_dataset = chainer.datasets.split_dataset(
        dataset, len(dataset) - 500)

    encoder = rationale.models.Encoder(
        w2v.shape[1], order, 200, 2, dropout=0.1
    )
    generator_cls = (rationale.models.GeneratorDependent
                     if dependent else rationale.models.Generator)
    # Original impl. uses two layers to model bi-directional LSTM
    generator = generator_cls(w2v.shape[1], order, 200, dropout=0.1)
    model = rationale.models.RationalizedRegressor(
        generator, encoder, w2v.shape[0], w2v.shape[1], initialEmb=w2v,
        dropout_emb=0.1, fix_embedding=fix_embedding,
        sparsity_coef=sparsity_coef, coherent_coef=coherent_coef
    )

    if gpu >= 0:
        logger.info('Using GPU (%d)' % gpu)
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()  # Copy the model to the GPU
    elif chainer.backends.intel64.is_ideep_available():
        logger.info('Using CPU with iDeep')
        # iDeep was able to accelerate training on CPU by about 30% on laptop
        model.to_intel64()
        chainer.global_config.use_ideep = 'auto'
    else:
        logger.info('Using CPU without acceleration')

    # Impl. by author uses mean as loss. Let's divide lr by batchsize to have
    # similar effect
    optimizer = chainer.optimizers.Adam(alpha=lr / batchsize)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(3.0))
    l2_reg = 1e-7
    # Impl. by author implements Weight decay as L2 loss, thus multiplying it
    # by the learning rate. Let's implement it that way.
    optimizer.add_hook(chainer.optimizer.WeightDecay(l2_reg * lr))

    train_iter = chainer.iterators.SerialIterator(train_dataset, batchsize)

    # Set up a trainer
    updater = training.StandardUpdater(
        train_iter, optimizer, device=gpu,
        converter=rationale.training.convert)

    trainer = training.Trainer(updater, (epoch, 'epoch'), out=out)

    logger.info("train: {},  dev: {}".format(
        len(train_dataset), len(dev_dataset)))
    # Evaluate the model with the development dataset for each epoch
    dev_iter = chainer.iterators.SerialIterator(
        dev_dataset, batchsize, repeat=False, shuffle=False)

    evaluator = extensions.Evaluator(
        dev_iter, model, device=gpu,
        converter=rationale.training.convert)
    trainer.extend(evaluator, trigger=frequency)

    inv_vocab = {v: k for k, v in vocab.items()}

    @chainer.training.make_extension()
    def monitor_rationale(_):
        batch = dev_dataset[np.random.choice(len(dev_dataset))]
        batch = rationale.training.convert([batch], gpu)
        z = chainer.cuda.to_cpu(model.predict_rationale(batch['xs'])[0])
        source = [inv_vocab[int(xi)] for xi in chainer.cuda.to_cpu(batch['xs'][0])]
        result = [t if zi > 0.5 else '_' for t, zi in zip(source, z)]
        print('# source : ' + ' '.join(source))
        print('# result : ' + ' '.join(result))

    trainer.extend(monitor_rationale, trigger=(10, 'iteration'))
    trainer.extend(
        SaveRestore(filename='trainer.npz'),
        trigger=MinValueTrigger('validation/main/generator/cost'),
        priority=96)

    trainer.extend(
        ConditionalRestart(
            monitor='validation/main/generator/cost', mode='min',
            patients=2))


    if gpu < 0:
        # ParameterStatistics does not work with GPU as of chainer 2.x
        # https://github.com/chainer/chainer/issues/3027
        trainer.extend(extensions.ParameterStatistics(
            model, trigger=(100, 'iteration')), priority=99)

    # Write a log of evaluation statistics for each iteration
    trainer.extend(extensions.LogReport(trigger=(1, 'iteration')), priority=98)
    trainer.extend(
        extensions.PrintReport(
            ['epoch', 'main/encoder/mse', 'main/generator/cost',
             'validation/main/encoder/mse', 'validation/main/generator/cost'],
            log_report=extensions.LogReport(trigger=(10, 'iteration'))),
        trigger=(10, 'iteration'), priority=97)

    if resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(resume, trainer)

    logger.info("Started training")
    trainer.run()

    # Save final model (without trainer)
    chainer.serializers.save_npz(os.path.join(out, 'trained_model.npz'), model)
    with open(os.path.join(out, 'vocab.json'), 'w') as fout:
        json.dump(vocab, fout)


if __name__ == '__main__':
    run()
