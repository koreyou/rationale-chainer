import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import six
from chainer import reporter

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


def select_tokens(x, z):
    """

    Args:
        x (list of Variable): Each variable is of shape (sequence, features)
        z (list of Variable): Each variable is of shape (sequence, 1)

    Returns:
        list of numpy.ndarray or cupy.ndarray

    """
    # sample from binomial distribution regarding z as probability
    return [xi.data[np.random.binomial(np.ones(zi.shape, dtype=np.int32), zi.data)]
            for xi, zi in zip(x, z)]


class RationalizedRegressor(chainer.Chain):

    def __init__(self, generator, encoder, n_vocab, emb_size, initialEmb=None,
                 dropout_emb=0.1, fix_embedding=False):
        super(RationalizedRegressor, self).__init__()
        if initialEmb is None: initialEmb = embed_init
        self.dropout_emb = dropout_emb
        self.fix_embedding = fix_embedding
        with self.init_scope():
            self.embed = L.EmbedID(
                n_vocab, emb_size, initialW=initialEmb, ignore_label=-1)
            self.encoder = encoder
            self.generator = generator

    def __call__(self, xs, ys):
        pred, z = self._forward(xs)
        # calculate loss for encoder
        loss_encoder = F.mean_squared_error(pred, ys)
        reporter.report({'encoder/loss': loss_encoder}, self)

        # calculate loss for generator
        sparsity_cost = self.xp.array([self.xp.sum(zi.data) for zi in z])
        conherence_cost = self.xp.array(
            [np.linalg.norm(zi.data[:-1]- zi.data[1:]) for zi in z])
        cost = (pred.data - ys) ** 2 + sparsity_cost + conherence_cost
        loss_generator = cost * F.log(F.stack(map(F.sum, z)))
        reporter.report({'generator/cost': self.xp.sum(cost)}, self)

        loss = loss_encoder + F.sum(loss_generator)
        reporter.report({'loss': loss.data}, self)
        return loss

    def _forward(self, xs):
        exs = sequence_embed(self.embed, xs, self.dropout_emb)
        if self.fix_embedding:
            for i in six.moves.xrange(len(exs)):
                exs[i].unchain_backward()
        z = self.generator(exs)
        xs_selected = select_tokens(exs, z)
        y = self.encoder(xs_selected)
        return y, z

    def predict(self, xs):
        y, _ = self._forward(xs)
        return y.data
