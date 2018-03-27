# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import json
import logging
import os

import chainer
import dill  # This is for joblib to use dill. Do NOT delete it.
import click
from chainer import training
from chainer.training import extensions
from joblib import Memory

import rationale
from rationale.training import SaveRestore, EarlyStoppingTrigger
from rationale.models import LSTMGenerator, LSTMEncoder, RationalizedRegressor
from rationale.dataset import prepare_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.argument('aspect', type=int)
@click.argument('train', type=click.Path(exists=True))
@click.argument('word2vec', type=click.Path(exists=True))
@click.option('--epoch', '-e', type=int, default=15,
              help='Number of sweeps over the dataset to train')
@click.option('--frequency', '-f', default=[1, 'epoch'],
              type=(int, click.Choice(['epoch', 'iteration'])),
              help='Frequency of taking a snapshot')
@click.option('--gpu', '-g', type=int, default=-1,
              help='GPU ID (negative value indicates CPU)')
@click.option('--out', '-o', default='result',
              help='Directory to output the result and temporaly file')
@click.option('--test', '-o', default=None, type=click.Path(exists=True))
@click.option('--batchsize', '-b', type=int, default=300,
              help='Number of images in each mini-batch')
@click.option('--lr', type=float, default=0.001, help='Learning rate')
@click.option('--fix_embedding', type=bool, default=False,
              help='Fix word embedding during training')
@click.option('--resume', '-r', default='',
              help='Resume the training from snapshot')
def run(aspect, train, word2vec, epoch, frequency, gpu, out, test, batchsize, lr,
        fix_embedding, resume):
    """
    Train "Rationalizing Neural Predictions" for one specified aspect.

    Please refer README.md for details.
    """
    memory = Memory(cachedir=out, verbose=1)
    w2v, vocab, dataset, test_dataset = \
        memory.cache(prepare_data)(train, word2vec, aspect, test_path=test)
    train_dataset, dev_dataset = chainer.datasets.split_dataset(dataset, -500)

    encoder = rationale.models.LSTMEncoder(
        1, 300,
    )
    generator = rationale.models.LSTMGenerator(
        2, 300, dropout=0.1
    )
    model = rationale.models.RationalizedRegressor(
        generator, encoder, w2v.shape[0], w2v.shape[1], initialEmb=w2v,
        dropout_emb=0.1, fix_embedding=fix_embedding)

    if gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam(alpha=lr)
    optimizer.setup(model)

    train_iter = chainer.iterators.SerialIterator(train_dataset, batchsize)

    # Set up a trainer
    updater = training.StandardUpdater(
        train_iter, optimizer, device=gpu,
        converter=rationale.training.convert)

    # Set up dev dataset
    stop_trigger = EarlyStoppingTrigger(
        monitor='validation/main/loss', max_trigger=(epoch, 'epoch'))
    trainer = training.Trainer(updater, stop_trigger, out=out)

    logger.info("train: {},  dev: {}".format(
        len(train_dataset), len(dev_dataset)))
    # Evaluate the model with the development dataset for each epoch
    dev_iter = chainer.iterators.SerialIterator(
        dev_dataset, batchsize, repeat=False, shuffle=False)

    evaluator = extensions.Evaluator(
        dev_iter, model, device=gpu,
        converter=rationale.training.convert)
    trainer.extend(evaluator, trigger=frequency)
    # This works together with EarlyStoppingTrigger to provide more reliable
    # early stopping
    trainer.extend(
        SaveRestore(),
        trigger=chainer.training.triggers.MinValueTrigger(
            'validation/main/loss'))

    # Take a snapshot for each specified epoch
    if gpu < 0:
        # ParameterStatistics does not work with GPU as of chainer 2.x
        # https://github.com/chainer/chainer/issues/3027
        trainer.extend(extensions.ParameterStatistics(
            model, trigger=(100, 'iteration')), priority=99)

    # Write a log of evaluation statistics for each iteration
    trainer.extend(extensions.LogReport(trigger=(1, 'iteration')), priority=98)
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy',
         'validation/main/accuracy']), trigger=frequency, priority=97)

    if resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(resume, trainer)

    logger.info("Started training")
    trainer.run()

    # Save final model (without trainer)
    chainer.serializers.save_npz(os.path.join(out, 'trained_model'), model)
    with open(os.path.join(out, 'vocab.json'), 'wb') as fout:
        json.dump(vocab, fout)


if __name__ == '__main__':
    run()