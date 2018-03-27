import chainer
import chainer.functions as F
import chainer.links as L


class LSTMGenerator(chainer.Chain):
    def __init__(self, n_layers, n_units, dropout=0., dropout_fc=0.1):
        super(LSTMGenerator, self).__init__()
        self.dropout_fc = dropout_fc
        self.out_units = n_units
        with self.init_scope():
            self.encoder = L.NStepLSTM(n_layers, n_units, n_units, dropout)
            self.l1 = L.Linear(n_units.out_units, 1)

    def __call__(self, x):
        # o: list of array of shape (sequence size, feature size)
        _, _, o = self.encoder(None, None, x)
        if self.dropout_fc > 0.:
            o = [F.dropout(o_i, self.dropout_fc) for o_i in o]
        o = [F.sigmoid(self.l1(o_i)) for o_i in o]
        return o