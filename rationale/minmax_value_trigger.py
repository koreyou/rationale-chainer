# -*- coding: utf-8 -*-
"""
This file is adopted from Chainer official implementation with small
modifications.
https://github.com/chainer/chainer/blob/v4.2.0/chainer/training/triggers/minmax_value_trigger.py
"""


class BestValueTrigger(object):

    """Trigger invoked when specific value becomes best. This will run every
    time key value is observed.

    Args:
        key (str): Key of value.
        compare (callable): Compare function which takes current best value and
            new value and returns whether new value is better than current
            best.
    """

    def __init__(self, key, compare):
        self._key = key
        self._best_value = None
        self._compare = compare

    def __call__(self, trainer):
        """Decides whether the extension should be called on this iteration.
        Args:
            trainer (~chainer.training.Trainer): Trainer object that this
                trigger is associated with. The ``observation`` of this trainer
                is used to determine if the trigger should fire.
        Returns:
            bool: ``True`` if the corresponding extension should be invoked in
            this iteration.
        """

        observation = trainer.observation
        key = self._key
        if key not in observation.keys():
            return False
        value = float(observation[key])  # copy to CPU
        if self._best_value is None or self._compare(self._best_value, value):
            self._best_value = value
            return True
        return False


class MaxValueTrigger(BestValueTrigger):

    """Trigger invoked when specific value becomes maximum.
    This will run every time key value is observed.
    Args:
        key (str): Key of value. The trigger fires when the value associated
            with this key becomes maximum.
    """

    def __init__(self, key):
        super(MaxValueTrigger, self).__init__(
            key, lambda max_value, new_value: new_value > max_value)


class MinValueTrigger(BestValueTrigger):

    """Trigger invoked when specific value becomes minimum.
    This will run every time key value is observed.
    Args:
        key (str): Key of value. The trigger fires when the value associated
            with this key becomes minimum.
    """

    def __init__(self, key):
        super(MinValueTrigger, self).__init__(
            key, lambda min_value, new_value: new_value < min_value)