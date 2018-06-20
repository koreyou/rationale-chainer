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

    def __call__(self, x):
        last_hs = []
        for i in range(self.n_layer):
            if self.dropout > 0.:
                x = [F.dropout(xi, self.dropout) for xi in x]
            encoder = getattr(self, "encoder%d" % i)
            x, _ = encoder(x)
            last_hs.append(F.stack([xi[-1] for xi in x]))
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
