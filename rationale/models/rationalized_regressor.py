import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import six
from chainer import reporter

embed_init = chainer.initializers.Uniform(.25)


def sparsity_cost(z):
    xp = chainer.cuda.get_array_module(z[0])
    return xp.stack([xp.sum(zi) for zi in z])

def coherence_cost(z):
    xp = chainer.cuda.get_array_module(z[0])
    z = [zi.astype(np.float32) for zi in z]
    return xp.stack([xp.sum(xp.abs(zi[:-1] - zi[1:])) for zi in z])


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
        pred, z_prob, z = self.forward(xs)
        loss = self.calc_loss(pred, z, z_prob, ys)[0]
        return F.average(loss)

    def calc_loss(self, pred, z, z_prob, ys):
        """

        Args:
            pred (chainer.Variable): Predicted score
            z (list of numpy.ndarray):
            ys (chainer.Variable): Ground truth score

        Returns:

        """
        xp = self.xp

        # calculate loss for encoder
        loss_encoder = (pred - ys) ** 2
        reporter.report({'encoder/loss': xp.average(loss_encoder.data)}, self)

        # calculate loss for generator
        sparsity = sparsity_cost(z)
        reporter.report({'generator/sparsity_cost': xp.average(sparsity)}, self)
        coherence = coherence_cost(z)
        reporter.report({'generator/conherence_cost': xp.average(coherence)}, self)
        regressor_cost = loss_encoder.data
        reporter.report({'generator/regressor_cost': xp.average(regressor_cost)}, self)
        cost = (regressor_cost +
                self.sparsity_coef * sparsity +
                self.coherent_coef * coherence)
        logpz = - F.stack(
            [F.sum(zi * F.log(p) + (1. - zi) * F.log(1. - p))
             for zi, p in zip(z, z_prob)]
        )
        loss_generator = cost * logpz
        reporter.report({'generator/cost': xp.average(cost)}, self)
        reporter.report({'generator/loss': xp.average(loss_generator.data)}, self)

        loss = loss_encoder + loss_generator
        reporter.report({'loss': xp.average(loss.data)}, self)
        return loss, loss_encoder, sparsity, coherence, regressor_cost, loss_generator

    def forward(self, xs):
        xp = self.xp
        exs = sequence_embed(self.embed, xs, self.dropout_emb)
        if self.fix_embedding:
            for i in six.moves.xrange(len(exs)):
                exs[i].unchain_backward()
        z = self.generator(exs)
        if xp.isnan(xp.sum(xp.stack([xp.sum(zi.data) for zi in z]))):
            raise ValueError("NaN detected in forward operation of generator")
        z_selected = self._sample(z)
        exs_selected = [ex[zi] for ex, zi in zip(exs, z_selected)]
        y = self.encoder(exs_selected)
        if xp.isnan(xp.sum(y.data)):
            raise ValueError("NaN detected in forward operation of encoder")
        return y, z, z_selected

    def _sample_prob(self, zi):
        """
        sample from binomial distribution regarding z as probability

        Args:
            zi:

        Returns:

        """
        selection = self.xp.array([False])
        # sample for as many as needed to get at least one token
        while not self.xp.any(selection):
            selection = self.xp.random.rand(*zi.shape) < zi
        return selection

    def _sample_deterministic(self, zi):
        selection = 0.5 < zi
        if not self.xp.any(selection):
            selection = np.zeros_like(zi)
            selection[np.argmax(zi):np.argmax(zi) + 1] = 1.
            return selection
        else:
            return selection

    def _sample(self, z):
        """

        Args:
            z (list of Variable): Each variable is of shape (sequence, 1)

        Returns:
            list of numpy.ndarray or cupy.ndarray

        """
        sample_func = self._sample_prob if chainer.config.train else self._sample_deterministic
        return [sample_func(zi.data) for zi in z]

    def predict(self, xs):
        y, _, z = self.forward(xs)
        return y.data, z

    def predict_class(self, xs):
        y, _, _ = self.forward(xs)
        return y.data

    def predict_rationale(self, xs):
        _, _, z = self.forward(xs)
        return z
