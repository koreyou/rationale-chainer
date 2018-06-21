import chainer
import chainer.functions as F
import chainer.links as L
from chainer import reporter

from rationale.models.dependent_selection import DependentSelectionLayer
from rationale.models.rcnn import BiRCNN


class Generator(chainer.Chain):
    """
    Generator that extract a rationale (a subset of text) from input.
    It samples rationales with randomness when in train mode. It picks the
    most likely rationales deterministically on test mode.

    This is the indepedent implementation of generator, i.e. it does not use
    sampled z_i when calculating z_{i+1}.
    """
    def __init__(self, in_size, order, n_units, dropout=0.1):
        super(Generator, self).__init__()
        self.dropout = dropout
        self.out_units = n_units
        with self.init_scope():
            self.encoder = BiRCNN(in_size, n_units, order)
            self.l1 = L.Linear(n_units * 2, 1)

    def __call__(self, x):
        xp = self.xp
        if self.dropout > 0.:
            x = [F.dropout(xi, self.dropout) for xi in x]
        # o: list of array of shape (sequence size, feature size)
        h, _ = self.encoder(x)
        if self.dropout > 0.:
            h = [F.dropout(h_i, self.dropout) for h_i in h]
        pz = [F.sigmoid(F.squeeze(self.l1(h_i), 1)) for h_i in h]
        z = self._sample(pz)
        selected_ratio = (xp.sum(xp.stack([xp.sum(zi) for zi in z]))
                          / xp.sum(xp.array([len(zi) for zi in z])))
        reporter.report({'generator/selected_ratio': selected_ratio}, self)

        return pz, z

    def _sample_prob(self, zi):
        """
        sample from binomial distribution regarding z as probability

        Args:
            zi (numpy.ndarray or cupy.ndarray): Probability vector

        Returns:
            numpy.ndarray or cupy.ndarray: Sampled boolean vector
        """
        return self.xp.random.rand(*zi.shape) < zi

    def _sample_deterministic(self, zi):
        """
        Sample deterministically regarding z as probability

        Args:
            zi (numpy.ndarray or cupy.ndarray): Probability vector

        Returns:
            numpy.ndarray or cupy.ndarray: Sampled boolean vector
        """
        return 0.5 < zi

    def _sample(self, z):
        """

        Args:
            z (list of Variable): Each variable is of shape (sequence, 1)

        Returns:
            list of numpy.ndarray or cupy.ndarray

        """
        sample_func = self._sample_prob if chainer.config.train else self._sample_deterministic
        return [sample_func(zi.data) for zi in z]


class GeneratorDependent(chainer.Chain):
    """
    Generator that extract a rationale (a subset of text) from input.
    It samples rationales with randomness when in train mode. It picks the
    most likely rationales deterministically on test mode.

    This is the depedent implementation of generator, i.e. it utilizes sampled
    z_i to calculating z_{i+1} with another RCNN.
    """
    def __init__(self, in_size, order, n_units, dropout=0.1):
        super(GeneratorDependent, self).__init__()
        self.dropout = dropout
        self.out_units = n_units
        with self.init_scope():
            self.encoder = BiRCNN(in_size, n_units, order)
            self.sampler = DependentSelectionLayer(n_units * 2, n_units)

    def __call__(self, x):
        xp = self.xp
        if self.dropout > 0.:
            x = [F.dropout(xi, self.dropout) for xi in x]
        # o: list of array of shape (sequence size, feature size)
        h, _ = self.encoder(x)
        if self.dropout > 0.:
            h = [F.dropout(h_i, self.dropout) for h_i in h]
        pz, z = self.sampler(h)

        selected_ratio = (xp.sum(xp.stack([xp.sum(zi) for zi in z]))
                          / xp.sum(xp.array([len(zi) for zi in z])))
        reporter.report({'generator/selected_ratio': selected_ratio}, self)

        return pz, z
