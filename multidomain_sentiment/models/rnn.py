import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import six

embed_init = chainer.initializers.Uniform(.25)


def sequence_embed(embed, xs, dropout=0.):
    """Efficient embedding function for variable-length sequences

    This output is equally to
    "return [F.dropout(embed(x), ratio=dropout) for x in xs]".
    However, calling the functions is one-shot and faster.

    Args:
        embed (callable): A :func:`~chainer.functions.embed_id` function
            or :class:`~chainer.links.EmbedID` link.
        xs (list of :class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): i-th element in the list is an input variable,
            which is a :math:`(L_i, )`-shaped int array.
        dropout (float): Dropout ratio.

    Returns:
        list of ~chainer.Variable: Output variables. i-th element in the
        list is an output variable, which is a :math:`(L_i, N)`-shaped
        float array. :math:`(N)` is the number of dimensions of word embedding.

    """
    xp = chainer.cuda.get_device_from_array(xs)
    x_len = [len(x) for x in xs]
    x_section = np.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    ex = F.dropout(ex, ratio=dropout)
    exs = F.split_axis(ex, x_section, 0)
    return exs


class RNNEncoder(chainer.Chain):

    """A LSTM-RNN Encoder with Word Embedding.

    This model encodes a sentence sequentially using LSTM.

    Args:
        n_layers (int): The number of LSTM layers.
        n_vocab (int): The size of vocabulary.
        n_units (int): The number of units of a LSTM layer and word embedding.
        dropout (float): The dropout ratio.

    """
    def __init__(self, n_layers, n_units, dropout=0.1):
        super(RNNEncoder, self).__init__()
        self.n_layers = n_layers
        self.out_units = n_units
        with self.init_scope():
            self.encoder = L.NStepLSTM(n_layers, n_units, n_units, dropout)

    def __call__(self, exs):
        last_h, last_c, ys = self.encoder(None, None, exs)
        assert(last_h.shape == (self.n_layers, len(exs), self.out_units))
        concat_outputs = last_h[-1]
        return concat_outputs


class MultiDomainRNNPredictor(chainer.Chain):
    def __init__(self, n_vocab, emb_size, encoders, shared_encoder, n_units,
                 n_class, initialEmb=None, dropout_emb=0.1,
                 dropout_fc=0.1, fix_embedding=False):
        super(MultiDomainRNNPredictor, self).__init__()
        if initialEmb is None: initialEmb = embed_init
        l1_in_units = encoders[0].out_units + shared_encoder.out_units
        self.dropout_emb = dropout_emb
        self.dropout_fc = dropout_fc
        self.fix_embedding = fix_embedding
        with self.init_scope():
            self.embed = L.EmbedID(
                n_vocab, emb_size, initialW=initialEmb, ignore_label=-1)
            self.encoders = encoders
            self.shared_encoder = shared_encoder
            self.l1 = L.Linear(l1_in_units, n_units)
            self.l2 = L.Linear(n_units, n_class)

    def __call__(self, xs, domains):
        assert self.xp.max(domains) < len(self.encoders)
        exs = sequence_embed(self.embed, xs, self.dropout_emb)
        if self.fix_embedding:
            for i in six.moves.xrange(len(exs)):
                exs[i].unchain_backward()
        o_e = [None] * len(xs)
        # FIXME: this is inefficient because it does not paralleize over domains
        for k, e in enumerate(self.encoders):
            domain_mask = np.where(domains == k)[0]
            if len(domain_mask) == 0:
                continue
            x_k = [exs[i] for i in domain_mask]
            y_k = e(x_k)
            y_k = F.split_axis(y_k, len(y_k), 0)
            for i, y in six.moves.zip(domain_mask, y_k):
                o_e[i] = y
        o_e = F.vstack(o_e)
        o_s = self.shared_encoder(exs)
        o = F.hstack((o_e, o_s))
        if self.dropout_fc > 0.:
            o = F.dropout(o, self.dropout_fc)
        o = F.tanh(self.l1(o))
        return self.l2(o)


def create_rnn_predictor(
        k, n_vocab, emb_size, fc_units, n_class, n_layers, rnn_units,
        dropout_rnn=0.1, initialEmb=None, dropout_emb=0.1, fix_embedding=False,
        dropout_fc=0.0):
    encoders = [RNNEncoder(n_layers, rnn_units, dropout=dropout_rnn)
                for _ in six.moves.xrange(k)]
    shared_encoder = RNNEncoder(n_layers, rnn_units, dropout=dropout_rnn)
    return MultiDomainRNNPredictor(
        n_vocab, emb_size, encoders, shared_encoder, fc_units, n_class,
        initialEmb=initialEmb, dropout_emb=dropout_emb, dropout_fc=dropout_fc,
        fix_embedding=fix_embedding)
