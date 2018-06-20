import chainer
import chainer.functions as F
import chainer.links as L

from rationale.models.rcnn import BiRCNN


class Generator(chainer.Chain):
    def __init__(self, in_size, order, n_units, dropout=0.1):
        super(Generator, self).__init__()
        self.dropout = dropout
        self.out_units = n_units
        with self.init_scope():
            self.encoder = BiRCNN(in_size, n_units, order)
            self.l1 = L.Linear(n_units * 2, 1)

    def __call__(self, x):
        if self.dropout > 0.:
            x = [F.dropout(xi, self.dropout) for xi in x]
        # o: list of array of shape (sequence size, feature size)
        o, _ = self.encoder(x)
        if self.dropout > 0.:
            o = [F.dropout(o_i, self.dropout) for o_i in o]
        o = [F.sigmoid(F.squeeze(self.l1(o_i), 1)) for o_i in o]
        return o
