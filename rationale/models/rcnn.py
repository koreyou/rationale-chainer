import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer.functions.array import transpose_sequence
from chainer.links.connection.n_step_rnn import argsort_list_descent, \
    permutate_list


class RCNN(chainer.Chain):
    def __init__(self, n_in, n_out, order):
        super(RCNN, self).__init__()
        self.order = order
        self.n_out = n_out
        with self.init_scope():
            self.forget_gate = L.Linear(n_in + n_out, 1)
            # bias is initialize with uniform distribution as in original impl.
            self.b = chainer.Parameter(
                initializer=chainer.initializers.Uniform(0.05),
                shape=[1, n_out])
        for i in range(order):
            self.add_param(
                "W%d" % i,
                shape=(n_in, n_out),
                initializer=chainer.initializers.Uniform(0.05))

    def _forward_single(self, x, h, c):
        """ forward one time step

        Args:
            x: Variable with shape (batch, n_features)
            h: Variable with shape (batch, n_out)
            c: Order-length list of Variable, each with shape (batch, n_out)

        Returns:

        """
        forget_ratio = F.sigmoid(self.forget_gate(F.hstack((x, h))))
        forget_ratio = F.broadcast_to(forget_ratio, c[0].shape)
        c_lst = []
        for i in range(self.order):
            W = getattr(self, "W%d" % i)
            c_i1_t1 = 0. if i == 0 else c[i - 1]
            c_i_t = forget_ratio * c[i] + (1. - forget_ratio) * (c_i1_t1 + F.matmul(x, W))
            c_lst.append(c_i_t)
        b = F.broadcast_to(self.b, c_i_t.shape)
        h_next = F.tanh(c_i_t + b)
        return h_next, c_lst

    def init_h(self, xs):
        shape = (len(xs), self.n_out)
        with cuda.get_device_from_id(self._device_id):
            h = chainer.Variable(self.xp.zeros(shape, dtype=xs[0].dtype))
        return h

    def init_c(self, xs):
        shape = (len(xs), self.n_out)
        with cuda.get_device_from_id(self._device_id):
            cs = [chainer.Variable(self.xp.zeros(shape, dtype=xs[0].dtype))
                  for _ in range(self.order)]
        return cs

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
        h = self.init_h(xs)
        c = self.init_c(xs)
        # list of Variable, each with shape (batch, features)
        trans_x = transpose_sequence.transpose_sequence(xs)

        c_lst = []
        h_lst = []
        for x in trans_x:
            if len(x) < len(h):
                # a sequence ended
                h = h[:len(x)]
                c = [c_i[:len(x)] for c_i in c]
            h, c = self._forward_single(x, h, c)
            c_lst.append(F.hstack(c))
            h_lst.append(h)

        # h_lst and c_lst basically have same order as x
        h_lst = transpose_sequence.transpose_sequence(h_lst)
        h_lst = permutate_list(h_lst, indices, inv=True)

        c_lst = transpose_sequence.transpose_sequence(c_lst)
        c_lst = permutate_list(c_lst, indices, inv=True)

        return h_lst, c_lst


class BiRCNN(chainer.Chain):
    def __init__(self, n_in, n_out, order):
        super(BiRCNN, self).__init__()
        with self.init_scope():
            self.fw = RCNN(n_in, n_out, order)
            self.bw = RCNN(n_in, n_out, order)

    def __call__(self, xs):
        """

        Args:
            xs (list or tuple): batch-length list of Variable, each with shape
                (

        Returns:

        """
        h_fw, c_fw = self.fw(xs)
        h_bw, c_bw = self.bw([x[::-1] for x in xs])
        h = [F.hstack((hi_fw, hi_bw[::-1])) for hi_fw, hi_bw in zip(h_fw, h_bw)]
        c = [F.hstack((ci_fw, ci_bw[::-1])) for ci_fw, ci_bw in zip(c_fw, c_bw)]
        return h, c
