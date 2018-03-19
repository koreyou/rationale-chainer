import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import six

embed_init = chainer.initializers.Uniform(.25)


def block_embed(embed, x, dropout=0.):
    """Embedding function followed by convolution
    Args:
        embed (callable): A :func:`~chainer.functions.embed_id` function
            or :class:`~chainer.links.EmbedID` link.
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable, which
            is a :math:`(B, L)`-shaped int array. Its first dimension
            :math:`(B)` is assumed to be the *minibatch dimension*.
            The second dimension :math:`(L)` is the length of padded
            sentences.
        dropout (float): Dropout ratio.
    Returns:
        ~chainer.Variable: Output variable. A float array with shape
        of :math:`(B, N, L, 1)`. :math:`(N)` is the number of dimensions
        of word embedding.
    """
    e = embed(x)
    e = F.dropout(e, ratio=dropout)
    e = F.transpose(e, (0, 2, 1))
    e = e[:, :, :, None]
    return e


class CNNEncoder(chainer.Chain):

    """A CNN encoder.
    This model encodes a sentence as a set of n-gram chunks
    using convolutional filters.
    Following the convolution, max-pooling is applied over time.

    Args:
        n_units (int): The number of units of MLP and word embedding.
        dropout (float): The dropout ratio.
    """

    def __init__(self, n_units):
        super(CNNEncoder, self).__init__()
        with self.init_scope():
            self.cnn_w3 = L.Convolution2D(
                None, n_units, ksize=(3, 1), stride=1, pad=(2, 0),
                nobias=True)
            self.cnn_w4 = L.Convolution2D(
                None, n_units, ksize=(4, 1), stride=1, pad=(3, 0),
                nobias=True)
            self.cnn_w5 = L.Convolution2D(
                None, n_units, ksize=(5, 1), stride=1, pad=(4, 0),
                nobias=True)
            self.out_units = n_units * 3

    def __call__(self, exs):
        h_w3 = F.max(self.cnn_w3(exs), axis=2)
        h_w4 = F.max(self.cnn_w4(exs), axis=2)
        h_w5 = F.max(self.cnn_w5(exs), axis=2)
        h = F.concat([h_w3, h_w4, h_w5], axis=1)
        h = F.relu(h)
        return h


class MultiDomainCNNPredictor(chainer.Chain):
    def __init__(self, n_vocab, emb_size, encoders, shared_encoder, n_units,
                 n_class, initialEmb=None, dropout_emb=0.1,
                 dropout_fc=0.1, fix_embedding=False):
        super(MultiDomainCNNPredictor, self).__init__()
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
        xs = chainer.dataset.convert.concat_examples(xs, padding=-1)
        exs = block_embed(self.embed, xs, self.dropout_emb)
        if self.fix_embedding:
            exs.unchain_backward()
        o_e = [None] * len(xs)
        # FIXME: this is inefficient because it does not paralleize over domains
        for k, e in enumerate(self.encoders):
            domain_mask = np.where(domains == k)[0]
            if len(domain_mask) == 0:
                continue
            x_k = F.get_item(exs, domain_mask)
            y_k = e(x_k)
            y_k = F.split_axis(y_k, len(y_k), 0)
            for i, y in six.moves.zip(domain_mask, y_k):
                o_e[i] = y
        o_e = F.vstack(o_e)
        o_s = self.shared_encoder(exs)
        o = F.hstack((o_e, o_s))
        if self.dropout_fc > 0.:
            o = F.dropout(o, self.dropout_fc)
        o = F.relu(self.l1(o))
        return self.l2(o)


def create_cnn_predictor(
        k, n_vocab, emb_size, fc_units, n_class, cnn_units, dropout_fc=0.1,
        initialEmb=None, dropout_emb=0.1, fix_embedding=False):
    encoders = [CNNEncoder(cnn_units // 3) for _ in six.moves.xrange(k)]
    shared_encoder = CNNEncoder(cnn_units // 3)
    return MultiDomainCNNPredictor(
        n_vocab, emb_size, encoders, shared_encoder, fc_units, n_class,
        initialEmb=initialEmb, dropout_emb=dropout_emb,
        fix_embedding=fix_embedding, dropout_fc=dropout_fc)
