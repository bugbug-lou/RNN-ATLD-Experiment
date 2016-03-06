import pickle
import theano
import theano.tensor as T

from utils import *


class Model(object):
    def __init__(self, input, target, sample_size, layers):
        self.input = input
        self.target = target
        self.layers = layers
        self.sample_size = sample_size

    def output(self, x):
        for l in self.layers:
            x = l(x)
        return x

    def build(self, train):
        self.set_phase(train)
        costs = self.costs
        updates = self.updates
        return costs, updates

    @property
    def params(self):
        p = []
        for l in self.layers:
            try:
                p.extend(l.params)
            except AttributeError:
                pass
        return p

    def reset(self):
        for l in self.layers:
            try:
                l.reset()
            except AttributeError:
                pass

    def set_phase(self, train):
        for l in self.layers:
            try:
                l.set_phase(train)
            except AttributeError:
                pass

    @property
    def updates(self):
        upd = []
        for l in self.layers:
            try:
                upd.extend(l.updates)
            except AttributeError:
                pass
        return upd

    @property
    def costs(self):
        p = self.output(self.input)
        cost = CrossEntropy()(p, self.target.flatten())
        pp = T.exp(cost)
        scaled_cost = self.sample_size * cost
        return [scaled_cost, pp]

    def dump(self, filename):
        params = []
        for i in xrange(len(self.params)):
            params.append(self.params[i].get_value())
        pickle.dump(params, open(filename, "wb"))

    def load(self, filename, strict=True):
        print "loading from %s" % filename
        params = pickle.load(open(filename, "rb"))
        self_params = self.params
        assert len(self_params) == len(params)
        for i in xrange(len(params)):
            if params[i].shape != self_params[i].get_value().shape:
                msg = self_params[i].name + ", model shape = " + str(self_params[i].get_value().shape) + \
                      " loading shape = " + str(params[i].shape)
                if strict:
                    raise Exception(msg)
                else:
                    print msg, "- skipping"
            else:
                self_params[i].set_value(params[i].astype(theano.config.floatX))

