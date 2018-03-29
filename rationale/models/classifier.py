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


class RationalizedRegressor(chainer.Chain):

    def __init__(self, generator, encoder, n_vocab, emb_size,
                 sparsity_coef=0.0003, coherent_coef=2.0, initialEmb=None,
                 dropout_emb=0.1, fix_embedding=False):
        super(RationalizedRegressor, self).__init__()
        if initialEmb is None: initialEmb = embed_init
        self.dropout_emb = dropout_emb
        self.fix_embedding = fix_embedding
        self.sparsity_coef = sparsity_coef
        self.coherent_coef = coherent_coef
        with self.init_scope():
            self.embed = L.EmbedID(
                n_vocab, emb_size, initialW=initialEmb, ignore_label=-1)
            self.encoder = encoder
            self.generator = generator

    def __call__(self, xs, ys):
        xp = self.xp
        pred, z = self._forward(xs)
        z_pred = [F.sigmoid(zi) for zi in z]
        # calculate loss for encoder
        loss_encoder = F.mean_squared_error(pred, ys)
        reporter.report({'encoder/loss': loss_encoder}, self)

        # calculate loss for generator
        sparsity_cost = xp.stack([xp.sum(zi.data) for zi in z_pred])
        reporter.report({'generator/sparsity_cost': xp.sum(sparsity_cost)}, self)
        conherence_cost = xp.stack(
            [xp.linalg.norm(zi.data[:-1]- zi.data[1:]) for zi in z_pred])
        reporter.report({'generator/conherence_cost': xp.sum(conherence_cost)}, self)
        regressor_cost = (pred.data - ys) ** 2
        reporter.report({'generator/regressor_cost': xp.sum(regressor_cost)}, self)
        cost = (regressor_cost +
                self.sparsity_coef * sparsity_cost +
                self.coherent_coef * conherence_cost)
        # log(p(z|x)) = log(prod(p(z|x)))
        #             = log(prod(sigmoid(zi)))
        #             = sum(log(sigmoid(zi)))
        #             = sum(-log1p(e^-zi))
        gen_prob = F.stack([F.sum(-F.log1p(F.exp(-zi))) for zi in z])
        loss_generator = cost * gen_prob
        reporter.report({'generator/cost': xp.sum(cost)}, self)
        reporter.report({'generator/loss': xp.sum(loss_generator.data)}, self)

        loss = loss_encoder + F.sum(loss_generator)
        reporter.report({'loss': loss.data}, self)
        return loss

    def _forward(self, xs):
        xp = self.xp
        exs = sequence_embed(self.embed, xs, self.dropout_emb)
        if self.fix_embedding:
            for i in six.moves.xrange(len(exs)):
                exs[i].unchain_backward()
        z = self.generator(exs)
        if xp.isnan(xp.sum(xp.stack([xp.sum(zi.data) for zi in z]))):
            raise ValueError("NaN detected in forward operation of generator")
        # we apply sigmoid here to avoid numerical instability in cost
        exs_selected = self.select_tokens(exs, [F.sigmoid(zi) for zi in z])
        y = self.encoder(exs_selected)
        if xp.isnan(xp.sum(y.data)):
            raise ValueError("NaN detected in forward operation of encoder")
        return y, z

    def select_tokens(self, x, z):
        """

        Args:
            x (list of Variable): Each variable is of shape (sequence, features)
            z (list of Variable): Each variable is of shape (sequence, 1)

        Returns:
            list of numpy.ndarray or cupy.ndarray

        """
        # sample from binomial distribution regarding z as probability
        return [xi[self.xp.random.rand(*zi.shape) < zi.data]
                for xi, zi in zip(x, z)]

    def predict(self, xs):
        y, _ = self._forward(xs)
        return y.data
