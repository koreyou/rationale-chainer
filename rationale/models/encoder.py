import chainer
import chainer.functions as F
import chainer.links as L


class BiLSTM(chainer.Chain):

    """A modified LSTM-RNN that used in Tao et al.

    Args:
        n_layers (int): The number of LSTM layers.
        n_units (int): The number of units of a LSTM layer and word embedding.
        dropout (float): The dropout ratio.
        use_all (bool): Whether to use outputs of all layers (by concatenating
            them all) or just the last layer.
    """
    def __init__(self, in_size, n_layers, n_units, dropout=0.1, use_all=True):
        super(BiLSTM, self).__init__()
        with self.init_scope():
            self.use_all = use_all
            self.encoder = L.NStepBiLSTM(n_layers, in_size, n_units, dropout)

    def __call__(self, exs):
        # last h has shape (2 * n_layers, batch size, n_units)
        last_h, _, _ = self.encoder(None, None, exs)
        if self.use_all:
            concat_outputs = last_h
        else:
            concat_outputs = last_h[-2:]
        concat_outputs = F.swapaxes(concat_outputs, 0, 1)
        concat_outputs = F.reshape(concat_outputs, (concat_outputs.shape[0], -1))
        return concat_outputs


class LSTMEncoder(chainer.Chain):

    def __init__(self, in_size, n_layers, n_units, dropout_rnn=0.1,
                 dropout_fc=0.1, use_all=True):
        super(LSTMEncoder, self).__init__()
        self.dropout_fc = dropout_fc
        if use_all:
            fc_units = n_layers * n_units * 2
        else:
            fc_units = n_units * 2
        with self.init_scope():
            self.encoder = BiLSTM(in_size, n_layers, n_units, dropout_rnn)
            self.l1 = L.Linear(fc_units, 1)

    def __call__(self, x):
        o = self.encoder(x)
        if self.dropout_fc > 0.:
            o = F.dropout(o, self.dropout_fc)
        return F.sigmoid(F.squeeze(self.l1(o), 1))
