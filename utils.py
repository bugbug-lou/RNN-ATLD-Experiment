import numpy as np
import theano
import theano.tensor as T


class Uniform(object):

    def __init__(self, scale=0.1):
        self.scale = scale

    def __call__(self, shape):
        return np.asarray(np.random.uniform(-self.scale, self.scale, shape), dtype=theano.config.floatX)


class Softmax(object):

    def __call__(self, x):
        return T.nnet.softmax(x.reshape((-1, x.shape[-1])))


class Identity(object):

    def __call__(self, x):
        return x


class CrossEntropy(object):

    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon

    def __call__(self, x, y):
        return T.nnet.categorical_crossentropy(x + self.epsilon, y).mean()


def ones(shape):
    return np.ones(shape).astype(theano.config.floatX)
