import chainer
import chainer.functions as F
import chainer.links as L

from rationale.models.rcnn import BiRCNN


class NStepBiRCNN(chainer.Chain):
    def __init__(self, n_in, n_out, order, n_layer, dropout=0.1):
        super(NStepBiRCNN, self).__init__()
        self.dropout = dropout
        self.n_layer = n_layer
        with self.init_scope():
            self.encoder0 = BiRCNN(n_in, n_out, order)
        for i in range(1, n_layer):
            self.add_link("encoder%d" % i, BiRCNN(n_out * 2, n_out, order))

    def _stack_and_fill(self, x, is_empty, feat_size):
        out = []
        x_cnt = 0
        for i in range(len(is_empty)):
            if is_empty[i]:
                out.append(self.xp.zeros([feat_size], self.xp.float32))
            else:
                out.append(x[x_cnt][-1])
                x_cnt += 1
        assert (len(is_empty) - sum(is_empty)) == x_cnt
        return out

    def __call__(self, x):
        last_hs = []
        # Generator may extract 0 token from the input, which causes an error
        # so only input non-zero length sequences to RCNN and fill those that
        # are empty
        is_empty = [len(xi) == 0 for xi in x]
        x = [x[i] for i in range(len(x)) if not is_empty[i]]
        if len(x) == 0:
            # all input is empty!
            shape = [len(is_empty), self.encoder0.n_out * self.n_layer]
            return chainer.Variable(self.xp.zeros(shape, self.xp.float32))
        for i in range(self.n_layer):
            if self.dropout > 0.:
                x = [F.dropout(xi, self.dropout) for xi in x]
            encoder = getattr(self, "encoder%d" % i)
            x, _ = encoder(x)
            last_hs.append(
                F.stack(self._stack_and_fill(x, is_empty, encoder.n_out)))
        return F.hstack(last_hs)


class Encoder(chainer.Chain):
    def __init__(self, in_size, order, n_units, n_layer, dropout=0.1):
        super(Encoder, self).__init__()
        self.dropout = dropout
        with self.init_scope():
            self.encoder = NStepBiRCNN(
                in_size, n_units, order, n_layer, dropout=dropout)
            self.l1 = L.Linear(n_layer * n_units * 2, 1)

    def __call__(self, x):
        o = self.encoder(x)
        if self.dropout > 0.:
            o = F.dropout(o, self.dropout)
        return F.sigmoid(F.squeeze(self.l1(o), 1))
