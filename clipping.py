import theano.tensor as T


class MaxNorm(object):

    def __init__(self, max_norm=5):
        self.max_norm = max_norm

    def __call__(self, grads):
        norm = T.sqrt(sum([T.sum(g ** 2) for g in grads]))
        return [self._clip(g, self.max_norm, norm) for g in grads]

    @staticmethod
    def _clip(g, c, n):
        if c > 0:
            g = T.switch(T.ge(n, c), g * c / n, g)
        return g


class Clip(object):

    def __init__(self, clip=5):
        self.clip = clip

    def __call__(self, grads):
        return [T.clip(g, -self.clip, self.clip) for g in grads]
