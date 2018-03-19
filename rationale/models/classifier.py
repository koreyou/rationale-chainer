import chainer
import chainer.functions as F
from chainer import reporter


class MultiDomainClassifier(chainer.Chain):

    def __init__(self, model, domain_dict=None):
        super(MultiDomainClassifier, self).__init__()
        with self.init_scope():
            self.model = model
            self.domain_dict = domain_dict

    def __call__(self, xs, ys, domains):
        concat_outputs = self.predict(xs, domains)
        loss = F.softmax_cross_entropy(concat_outputs, ys)
        accuracy = F.accuracy(concat_outputs, ys)
        reporter.report({'loss': loss.data}, self)
        reporter.report({'accuracy': accuracy.data}, self)
        output_cpu = chainer.cuda.to_cpu(concat_outputs.data)
        ys_cpu = chainer.cuda.to_cpu(ys)
        for k in set(domains):
            domain_mask = domains == k
            accuracy = F.accuracy(output_cpu[domain_mask], ys_cpu[domain_mask])
            if self.domain_dict is not None:
                name = 'accuracy_%s' % self.domain_dict[k]
            else:
                name = 'accuracy_%d' % k
            reporter.report({name: accuracy.data}, self)
        return loss

    def predict(self, xs, domains, softmax=False, argmax=False):
        o = self.model(xs, domains)
        if softmax:
            return F.softmax(o).data
        elif argmax:
            return self.xp.argmax(o.data, axis=1)
        else:
            return o
