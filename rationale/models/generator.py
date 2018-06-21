import chainer
import chainer.functions as F
import chainer.links as L

from rationale.models.rcnn import BiRCNN
from chainer import reporter

class Generator(chainer.Chain):
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
        pz, _ = self.encoder(x)
        if self.dropout > 0.:
            pz = [F.dropout(pz_i, self.dropout) for pz_i in pz]
        pz = [F.sigmoid(F.squeeze(self.l1(pz_i), 1)) for pz_i in pz]
        z = self._sample(pz)
        selected_ratio = (xp.sum(xp.stack([xp.sum(zi) for zi in z]))
                          / xp.sum(xp.array([len(zi) for zi in z])))
        reporter.report({'generator/selected_ratio': selected_ratio}, self)

        return pz, z

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
            selection = self.xp.zeros_like(zi)
            selection[self.xp.argmax(zi):self.xp.argmax(zi) + 1] = 1.
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
