# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import operator
import os
import random
import shutil
import tempfile
import warnings

import chainer
import numpy
import six
from chainer import cuda
from chainer import reporter
from chainer.training import util


def convert(batch, device):
    def to_device_batch_seq(batch):
        if device is None:
            return batch
        elif device < 0:
            return [chainer.dataset.to_device(device, x) for x in batch]
        else:
            xp = cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = numpy.cumsum([len(x) for x in batch[:-1]], dtype='i')
            concat_dev = chainer.dataset.to_device(device, concat)
            batch_dev = cuda.cupy.split(concat_dev, sections)
            return batch_dev

    def to_device_batch(batch):
        if device is None:
            return numpy.array(batch)
        elif device < 0:
            batch = numpy.array(batch)
            return chainer.dataset.to_device(device, batch)
        else:
            xp = cuda.cupy.get_array_module(*batch)
            return chainer.dataset.to_device(device, xp.array(batch))

    return {'xs': to_device_batch_seq([b['xs'] for b in batch]),
            'ys': to_device_batch([b['ys'] for b in batch])}


def _snapshot_object(_, target, filename, savefun):
    fd, tmppath = tempfile.mkstemp()
    try:
        savefun(tmppath, target)
    except Exception:
        os.close(fd)
        os.remove(tmppath)
        raise
    os.close(fd)
    shutil.move(tmppath, filename)


class SaveRestore(chainer.training.extension.Extension):

    """Trainer extension to save a snapshot and restore it at the end of
    training.

    Typical usage is:

    .. code-block:: python

        trainer.extend(
            SaveRestore(),
            trigger=chainer.training.triggers.MinValueTrigger('validation/main/loss'))

    which save will save snapshots and apply (pseudo-) early stopping by
    loading the snapshot with the best validation loss.

    Args:
        filename (str): Name of the file into which the object is serialized.
            It can be a format string, where the trainer object is passed to
            the :meth:`str.format` method. For example,
            ``'snapshot_{.updater.iteration}'`` is converted to
            ``'snapshot_10000'`` at the 10,000th iteration.
            Or you can give name without formatter, which will overwrite the
            saved object on each call, thus only keeping the best model on
            the disk.
            Or you can give None, in which case the object is saved to
            a temporaly path and deleted at the end of the training.
        savefun: Function to save the object. It takes two arguments: the
            output file path and the object to serialize.
        loadfun: Function to load the object. It takes two arguments: the
            file path and the object to deserialize.
    """
    priority = -100

    def __init__(self, filename='snapshot_iter_{.updater.iteration}',
                 savefun=chainer.serializers.npz.save_npz,
                 loadfun=chainer.serializers.npz.load_npz):
        super(SaveRestore, self).__init__()
        self._savefun = savefun
        self._loadfun = loadfun
        self._saved_iteration = None
        self._keep_snapshot = filename is not None
        self._filename = filename or 'saverestore' + str(hash(random.random()))

    def __call__(self, trainer):
        fn = self._filename.format(trainer)
        self._saved_path = os.path.join(trainer.out, fn)
        if not os.path.exists(trainer.out):
            os.makedirs(trainer.out)
        _snapshot_object(trainer, trainer, self._saved_path, self._savefun)
        self._saved_iteration = trainer.updater.iteration
        self._trainer = trainer  # get referencee to trainer

    def finalize(self):
        if self._saved_iteration is not None:
            print('Loading model from %d iteration' % self._saved_iteration)
            self._loadfun(self._saved_path, self._trainer)
        else:
            print('Warning: SaveRestore was never triggered')
        if not self._keep_snapshot and os.path.exists(self._saved_path):
            os.remove(self._saved_path)


class ConditionalRestart(chainer.training.extension.Extension):
    """Trainer extension to save a snapshot and restore it when certain condition
    are met.

    Args:
        monitor (str) : The metric you want to monitor
        check_trigger: Trigger that decides the comparison
            interval between current best value and new value.
            This must be a tuple in the form of ``<int>,
            'epoch'`` or ``<int>, 'iteration'`` which is passed to
            :class:`~chainer.training.triggers.IntervalTrigger`.
        filename (str): Name of the file into which the object is serialized.
            It can be a format string, where the trainer object is passed to
            the :meth:`str.format` method. For example,
            ``'snapshot_{.updater.iteration}'`` is converted to
            ``'snapshot_10000'`` at the 10,000th iteration.
            Or you can give name without formatter, which will overwrite the
            saved object on each call, thus only keeping the best model on
            the disk.
            Or you can give None, in which case the object is saved to
            a temporaly path and deleted at the end of the training.
        savefun: Function to save the object. It takes two arguments: the
            output file path and the object to serialize.
        loadfun: Function to load the object. It takes two arguments: the
            file path and the object to deserialize.
        onloadfun: Function to triggered after loading the object. Trainer
            object is passed as an argument. None to do nothing.
        patients (int) : Counts to let the trigger be patient.
            The trigger will not fire until the condition is met
            for successive ``patient`` checks.
        mode (str) : ``'max'`` or ``'min'``.
            It is used to determine how to compare the monitored values.

    """
    priority = 50  # let it have very low priority

    def __init__(
            self, monitor='main/loss',
            savefun=chainer.serializers.npz.save_npz,
            loadfun=chainer.serializers.npz.load_npz, onloadfun=None,
            check_trigger=(1, 'epoch'), patients=3, mode='min'
    ):
        super(ConditionalRestart, self).__init__()
        self._savefun = savefun
        self._loadfun = loadfun
        self._onloadfun = onloadfun
        self._saved_iteration = None
        self._filename = 'saverestore' + str(hash(random.random()))

        if mode == 'max':
            self._compare = operator.gt
            # never let max_trigger trigger
            self._load_trigger = chainer.training.triggers.EarlyStoppingTrigger(
                check_trigger=check_trigger, monitor=monitor, patients=patients,
                mode=mode, max_trigger=(1000000000, 'epoch')
            )
            self._save_trigger = chainer.training.triggers.MaxValueTrigger(
                monitor, trigger=check_trigger)
        elif mode == 'min':
            self._load_trigger = chainer.training.triggers.EarlyStoppingTrigger(
                check_trigger=check_trigger, monitor=monitor, patients=patients,
                mode=mode, max_trigger=(1000000000, 'epoch')
            )
            self._save_trigger = chainer.training.triggers.MinValueTrigger(
                monitor, trigger=check_trigger)
        else:
            raise ValueError('Unknown mode "%s"' % mode)

    def __call__(self, trainer):
        # be careful using triggers; they may have some side effect
        flag_load = self._load_trigger(trainer)
        flag_save = self._save_trigger(trainer)
        if self._saved_iteration is None:
            # Avoid it to be triggered on the very first call
            self._save(trainer)
        elif flag_load:
            self._load(trainer)
            if self._onloadfun is not None:
                self._onloadfun(trainer)
            self._load_trigger.count = 0
        elif flag_save:
            self._save(trainer)

    def _save(self, trainer):
        fn = self._filename.format(trainer)
        self._saved_path = os.path.join(trainer.out, fn)
        if not os.path.exists(trainer.out):
            os.makedirs(trainer.out)
        _snapshot_object(trainer, trainer, self._saved_path, self._savefun)
        self._saved_iteration = trainer.updater.iteration
        print('Saving model from %d iteration' % self._saved_iteration)

    def _load(self, trainer):
        print('Loading model from %d iteration' % self._saved_iteration)
        self._loadfun(self._saved_path, trainer)

    def finalize(self):
        if os.path.exists(self._saved_path):
            os.remove(self._saved_path)
