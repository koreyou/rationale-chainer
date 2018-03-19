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
            return xp.array(batch)

    keys = list(six.iterkeys(batch[0]))
    return {'xs': to_device_batch_seq([b['xs'] for b in batch]),
            'ys': to_device_batch([b['ys'] for b in batch]),
            'domains': to_device_batch([b['domains'] for b in batch])}


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


class EarlyStoppingTrigger(object):
    """Trigger for Early Stopping
    It can be used as a stop trigger of :class:`~chainer.training.Trainer`
    to realize *early stopping* technique.
    This trigger works as follows.
    Within each *check interval* defined by the ``check_trigger`` argument,
    it monitors and accumulates the reported value at each iteration.
    At the end of each interval, it computes the mean of the accumulated
    values and compares it to the previous ones to maintain the *best* value.
    When it finds that the best value is not updated
    for some periods (defined by `patients`), this trigger fires.

    Adopted from chainer v4.0.0b.

    Args:
        monitor (str) : The metric you want to monitor
        check_trigger: Trigger that decides the comparison
            interval between current best value and new value.
            This must be a tuple in the form of ``<int>,
            'epoch'`` or ``<int>, 'iteration'`` which is passed to
            :class:`~chainer.training.triggers.IntervalTrigger`.
        patients (int) : Counts to let the trigger be patient.
            The trigger will not fire until the condition is met
            for successive ``patient`` checks.
        mode (str) : ``'max'``, ``'min'``, or ``'auto'``.
            It is used to determine how to compare the monitored values.
        verbose (bool) : Enable verbose output.
            If verbose is true, you can get more information
        max_trigger (int) : Upper bound of the number of training loops
    """

    def __init__(self, check_trigger=(1, 'epoch'), monitor='main/loss',
                 patients=3, mode='auto', verbose=False,
                 max_trigger=(100, 'epoch')):

        self.count = 0
        self.patients = patients
        self.monitor = monitor
        self.verbose = verbose
        self.already_warning = False
        self._max_trigger = util.get_trigger(max_trigger)
        self._interval_trigger = util.get_trigger(check_trigger)

        self._init_summary()

        if mode == 'max':
            self._compare = operator.gt

        elif mode == 'min':
            self._compare = operator.lt

        else:
            if 'accuracy' in monitor:
                self._compare = operator.gt

            else:
                self._compare = operator.lt

        if self._compare == operator.gt:
            if verbose:
                print('early stopping: operator is greater')
            self.best = float('-inf')

        else:
            if verbose:
                print('early stopping: operator is less')
            self.best = float('inf')

    def __call__(self, trainer):
        """Decides whether the training loop should be stopped.
        Args:
            trainer (~chainer.training.Trainer): Trainer object that this
                trigger is associated with. The ``observation`` of this trainer
                is used to determine if the trigger should fire.
        Returns:
            bool: ``True`` if the training loop should be stopped.
        """

        observation = trainer.observation

        summary = self._summary

        if self.monitor in observation:
            summary.add({self.monitor: observation[self.monitor]})

        if self._max_trigger(trainer):
            return True

        if not self._interval_trigger(trainer):
            return False

        if self.monitor not in observation.keys():
            warnings.warn('{} is not in observation'.format(self.monitor))
            return False

        stat = self._summary.compute_mean()
        current_val = stat[self.monitor]
        self._init_summary()

        if self._compare(current_val, self.best):
            self.best = current_val
            self.count = 0

        else:
            self.count += 1

        if self._stop_condition():
            if self.verbose:
                print('Epoch {}: early stopping'.format(trainer.updater.epoch))
            return True

        return False

    def _stop_condition(self):
        return self.count >= self.patients

    def _init_summary(self):
        self._summary = reporter.DictSummary()
