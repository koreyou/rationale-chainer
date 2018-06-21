from itertools import chain

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.functions.array import transpose_sequence
from chainer.initializers import Uniform
from chainer.links.connection.n_step_rnn import argsort_list_descent, \
    permutate_list

from rationale.models.rcnn import RCNN


class DependentSelectionLayer(chainer.Chain):

    def __init__(self, n_in, n_hidden):
        super(DependentSelectionLayer, self).__init__()
        # input dim is incoming feature size (n_in) + previous selection z (1)
        with self.init_scope():
            self.rnn = RCNN(n_in + 1, n_hidden, 2)
            self.l = L.Linear(
                n_in + n_hidden, 1, initialW=Uniform(0.05),
                initial_bias=Uniform(0.05))

    def _forward_single(self, x, h_prev, c_prev):
        """ forward one time step

        Args:
            x: Variable with shape (batch, n_features)
            h: Variable with shape (batch, n_out)
            c: Order-length list of Variable, each with shape (batch, n_out)

        Returns:

        """
        # probability of choosing z
        pz = F.sigmoid(self.l(F.hstack((x, h_prev))))
        # sample (it is not differentiable)
        if chainer.config.train:
            z = self.xp.random.rand(*pz.shape) < pz.data
        else:
            z = 0.5 < pz.data

        xz = F.hstack((x, z.astype(self.xp.float32)))

        h, c = self.rnn.forward_single(xz, h_prev, c_prev)

        return pz, z, h, c

    def __call__(self, xs):
        """

        Args:
            xs (list or tuple): batch-length list of Variable, each with shape
                (

        Returns:

        """
        assert isinstance(xs, (list, tuple))
        indices = argsort_list_descent(xs)

        xs = permutate_list(xs, indices, inv=False)
        h = self.rnn.init_h(xs)
        c = self.rnn.init_c(xs)
        # list of Variable, each with shape (batch, features)
        trans_x = transpose_sequence.transpose_sequence(xs)

        z_lst = []
        pz_lst = []
        for x in trans_x:
            if len(x) < len(h):
                # a sequence ended
                h = h[:len(x)]
                c = [c_i[:len(x)] for c_i in c]
            pz, z, h, c = self._forward_single(x, h, c)
            z_lst.append(z)
            pz_lst.append(pz)

        # h_lst and c_lst basically have same order as x
        z_lst = transpose_sequence.transpose_sequence(z_lst)
        z_lst = permutate_list(z_lst, indices, inv=True)
        z_lst = [F.squeeze(z, 1).data for z in z_lst]

        # h_lst and c_lst basically have same order as x
        pz_lst = transpose_sequence.transpose_sequence(pz_lst)
        pz_lst = permutate_list(pz_lst, indices, inv=True)
        pz_lst = [F.squeeze(pz, 1) for pz in pz_lst]

        return pz_lst, z_lst
