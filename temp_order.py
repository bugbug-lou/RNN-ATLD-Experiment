import numpy
from random import randint
from random import uniform
import theano
import theano.tensor as T


class TempOrder:
    def __init__(self, low, high, length, batch_size=128, long_sequences=True):
        self.batch_size = batch_size
        self.input_size = 2
        self.output_size = 4
        self.length = length
        self.low = low
        self.high = high
        assert length > high
        assert high > low

    def get_batch(self):
        ins = numpy.zeros((self.batch_size, self.length, 2))
        outs = numpy.zeros((self.batch_size, 4))

        for i in xrange(self.batch_size):
            idx1 = randint(0, self.low)
            idx2 = randint(self.low, self.high)
            seq = numpy.random.binomial(1, 0.5, self.length)
            ins[i, :, 0] = seq
            ins[i, idx1, 1] = 1
            ins[i, idx2, 1] = 1

            label = ins[i, idx1, 0] * 2 + ins[i, idx2, 0]
            outs[i, label] = 1

        return ins, outs

    @staticmethod
    def costs(model_output, t):
        y = model_output[-1]
        s = T.exp(y) / T.sum(T.exp(y), axis=1, keepdims=True)
        cost = T.nnet.binary_crossentropy(s, t).mean()

        true_labels = T.argmax(t, axis=1)
        predicted_labels = T.argmax(y, axis=1)
        acc = T.mean(T.eq(true_labels, predicted_labels))

        return [cost, acc]
