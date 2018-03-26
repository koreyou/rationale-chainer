import chainer
import chainer.functions as F
import chainer.links as L


class LSTM(chainer.Chain):

    """A LSTM-RNN that outputs final state.

    Args:
        n_layers (int): The number of LSTM layers.
        n_units (int): The number of units of a LSTM layer and word embedding.
        dropout (float): The dropout ratio.
    """
    def __init__(self, n_layers, n_units, dropout=0.1):
        super(RNNEncoder, self).__init__()
        with self.init_scope():
            self.encoder = L.NStepLSTM(n_layers, n_units, n_units, dropout)

    def __call__(self, exs):
        last_h, _, _ = self.encoder(None, None, exs)
        concat_outputs = last_h[-1]
        return concat_outputs


class LSTMEncoder(chainer.Chain):

    def __init__(self, n_layers, n_units, dropout_rnn=0.1,
                 dropout_fc=0.1):
        super(LSTMEncoder, self).__init__()
        self.dropout_fc = dropout_fc
        with self.init_scope():
            self.encoder = LSTM(n_layers, n_units, dropout_rnn)
            self.l1 = L.Linear(n_units, 1)

    def __call__(self, x):
        o = self.encoder(x)
        if self.dropout_fc > 0.:
            o = F.dropout(o, self.dropout_fc)
        return self.l1(o)
