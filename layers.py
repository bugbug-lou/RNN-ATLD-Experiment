from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy
from utils import *

class Linear(object):

    def __init__(self, input_size, output_size, activation=Softmax(),
                 weight_init=Uniform()):
        self.W = theano.shared(weight_init((input_size, output_size)), name="W_linear")
        self.b = theano.shared(weight_init(output_size), name="b_linear")
        self.params = [self.W, self.b]

        self.activation = activation

    def __call__(self, x):
        return self.activation(T.dot(x, self.W) + self.b)


class Embed(object):
    def __init__(self, input_size, output_size, weight_init=Uniform(), learnable=True):
        self.W = theano.shared(weight_init((input_size, output_size)), name="W_emb")
        if learnable:
            self.params = [self.W]
        else:
            self.params = []

    def __call__(self, x):
        return self.W[x]


class Dropout(object):
    def __init__(self, p):
        self.srng = RandomStreams(seed=np.random.randint(1000000))
        self.p = p
        self.train = True

    def __call__(self, x):
        if self.train:
            return x * self.srng.binomial(x.shape, p=1-self.p, dtype=theano.config.floatX)/(1-self.p)
        return x

    def set_phase(self, train):
        self.train = train


class LSTM(object):
    def __init__(self, input_size, layer_size, batch_size, hid_dropout_rate, hid_scale, drop_candidates, per_step,
                 activation=T.tanh, inner_activation=T.nnet.sigmoid, weight_init=Uniform(), persistent=True):
        self.drop_candidates = drop_candidates
        self.per_step = per_step

        self.srng = RandomStreams(seed=np.random.randint(1000000))

        W_shape = input_size, layer_size
        U_shape = layer_size, layer_size
        b_shape = layer_size

        self.W_i = theano.shared(weight_init(W_shape), name="W_i")
        self.U_i = theano.shared(weight_init(U_shape), name="U_i")
        self.b_i = theano.shared(weight_init(b_shape), name="b_i")

        self.W_f = theano.shared(weight_init(W_shape), name="U_f")
        self.U_f = theano.shared(weight_init(U_shape), name="C_f")
        self.b_f = theano.shared(weight_init(b_shape), name="b_f")

        self.W_c = theano.shared(weight_init(W_shape), name="W_c")
        self.U_c = theano.shared(weight_init(U_shape), name="U_c")
        self.b_c = theano.shared(weight_init(b_shape), name="b_c")

        self.W_o = theano.shared(weight_init(W_shape), name="W_o")
        self.U_o = theano.shared(weight_init(U_shape), name="U_o")
        self.b_o = theano.shared(weight_init(b_shape), name="b_o")

        self.h = theano.shared(np.zeros((batch_size, layer_size), dtype=theano.config.floatX))
        self.c = theano.shared(np.zeros((batch_size, layer_size), dtype=theano.config.floatX))

        self.params = [
            self.W_i, self.U_i, self.b_i,
            self.W_c, self.U_c, self.b_c,
            self.W_f, self.U_f, self.b_f,
            self.W_o, self.U_o, self.b_o,
        ]

        self.activation = activation
        self.inner_activation = inner_activation

        self.updates = []
        self.train = True
        self.batch_size = batch_size
        self.hid_dropout_rate = hid_dropout_rate
        self.hid_scale = hid_scale
        self.layer_size = layer_size
        self.persistent = persistent

    def __call__(self, x):
        # (time_steps, batch_size, layers_size)
        xi, xf, xc, xo = self._input_to_hidden(x)

        if self.per_step:
            masks = self.srng.normal(loc = 0.0, scale = self.hid_scale, size = (1,self.batch_size,self.layer_size)) * self.srng.binomial(n=1, p=1-self.hid_dropout_rate, size=(xi.shape[0], self.batch_size, self.layer_size), dtype=theano.config.floatX)
            [outputs, memories], updates = theano.scan(self._hidden_to_hidden,
                sequences=[xi, xf, xo, xc, masks],
                outputs_info=[self.h, self.c],
                non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c],
            )
        else:
            tau = self.srng.normal(size = (self.batch_size,1), avg = 0.0, std = self.hid_scale, dtype=theano.config.floatX)
            masks = tau * tau * self.srng.binomial(n=1, p=1-self.hid_dropout_rate, size=(self.batch_size, self.layer_size), dtype=theano.config.floatX)
            [outputs, memories], updates = theano.scan(self._hidden_to_hidden_per_seq,
                sequences=[xi, xf, xo, xc],
                outputs_info=[self.h, self.c],
                non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c, mask],
            )

        if self.persistent:
            self.updates = [(self.h, outputs[-1]), (self.c, memories[-1])]

        return outputs.dimshuffle((1, 0, 2))

    def _input_to_hidden(self, x):
        # (time_steps, batch_size, input_size)
        x = x.dimshuffle((1, 0, 2))

        xi = T.dot(x, self.W_i) + self.b_i
        xf = T.dot(x, self.W_f) + self.b_f
        xc = T.dot(x, self.W_c) + self.b_c
        xo = T.dot(x, self.W_o) + self.b_o
        return xi, xf, xc, xo

    def _hidden_to_hidden_per_seq(self,
        xi_t, xf_t, xo_t, xc_t,
        h_tm1, c_tm1,
        u_i, u_f, u_o, u_c, mask):

        return self._hidden_to_hidden(xi_t, xf_t, xo_t, xc_t, mask, h_tm1, c_tm1, u_i, u_f, u_o, u_c)

    def _hidden_to_hidden(self,
        xi_t, xf_t, xo_t, xc_t, mask_t,
        h_tm1, c_tm1,
        u_i, u_f, u_o, u_c):

        i_t = self.inner_activation(xi_t + T.dot(h_tm1, u_i))
        cand_t = i_t * self.activation(xc_t + T.dot(h_tm1, u_c))

        if self.drop_candidates:
            cand_t = cand_t * mask_t if self.train else cand_t * (1 - self.hid_dropout_rate)

        f_t = self.inner_activation(xf_t + T.dot(h_tm1, u_f))
        c_t = f_t * c_tm1 + cand_t

        if not self.drop_candidates:
            c_t = c_t * mask_t if self.train else c_t * (1 - self.hid_dropout_rate)

        o_t = self.inner_activation(xo_t + T.dot(h_tm1, u_o))
        h_t = o_t * self.activation(c_t)
        return h_t, c_t

    def set_phase(self, train):
        self.train = train

    def reset(self):
        self.h = theano.shared(np.zeros((self.batch_size, self.layer_size), dtype=theano.config.floatX))
        self.c = theano.shared(np.zeros((self.batch_size, self.layer_size), dtype=theano.config.floatX))


class GRU(object):

    def __init__(self, input_size, layer_size, batch_size, hid_dropout_rate, drop_candidates, per_step,
                 activation=T.tanh, weight_init=Uniform()):
        self.srng = RandomStreams(seed=np.random.randint(1000000))
        self.drop_candidates = drop_candidates
        self.per_step = per_step

        self.W_r = theano.shared(weight_init((input_size, layer_size)), name='W_r')
        self.U_r = theano.shared(weight_init((layer_size, layer_size)), name='W_r')
        self.b_r = theano.shared(weight_init(layer_size), name='b_r')
        self.W_z = theano.shared(weight_init((input_size, layer_size)), name='W_z')
        self.U_z = theano.shared(weight_init((layer_size, layer_size)), name='W_z')
        self.b_z = theano.shared(weight_init(layer_size), name='b_z')
        self.W_h = theano.shared(weight_init((input_size, layer_size)), name='W_h')
        self.U_h = theano.shared(weight_init((layer_size, layer_size)), name='U_h')
        self.b_h = theano.shared(weight_init(layer_size), name="b_h")

        self.h = theano.shared(np.zeros((batch_size, layer_size), dtype=theano.config.floatX))

        self.params = [
            self.W_r, self.U_r, self.b_r,
            self.W_z, self.U_z, self.b_z,
            self.W_h, self.U_h, self.b_h
        ]

        self.activation = activation

        self.updates = []
        self.train = False
        self.layer_size = layer_size
        self.batch_size = batch_size
        self.hid_dropout_rate = hid_dropout_rate

    def __call__(self, x):
        # (time_steps, batch_size, layers_size)
        xr, xz, xh = self._input_to_hidden(x)

        if self.per_step:
            masks = self.srng.binomial(n=1, p=1-self.hid_dropout_rate, size=(xr.shape[0], self.batch_size, self.layer_size), dtype=theano.config.floatX)
            outputs, updates = theano.scan(self._hidden_to_hidden,
                sequences=[xr, xz, xh, masks],
                outputs_info=[self.h],
            )
        else:
            mask = self.srng.binomial(n=1, p=1-self.hid_dropout_rate, size=(self.batch_size, self.layer_size), dtype=theano.config.floatX)
            outputs, updates = theano.scan(self._hidden_to_hidden_per_seq,
                sequences=[xr, xz, xh],
                outputs_info=[self.h],
                non_sequences=[mask],
            )

        self.updates = [(self.h, outputs[-1])]

        return outputs.dimshuffle((1, 0, 2))

    def _input_to_hidden(self, x):
        x = x.dimshuffle((1, 0, 2))

        r = T.dot(x, self.W_r) + self.b_r
        z = T.dot(x, self.W_z) + self.b_z
        h = T.dot(x, self.W_h) + self.b_h

        return r, z, h

    def _hidden_to_hidden(self, xr_t, xz_t, xh_t, mask_t, h_tm1):
        r_t = T.nnet.sigmoid(xr_t + T.dot(h_tm1, self.U_r))
        z_t = T.nnet.sigmoid(xz_t + T.dot(h_tm1, self.U_z))
        cand_t = self.activation(xh_t + T.dot(r_t * h_tm1, self.U_h))
        if self.drop_candidates:
            if self.train:
                cand_t = cand_t * mask_t
            else:
                cand_t = cand_t * (1 - self.hid_dropout_rate)

        h_t = (1 - z_t) * h_tm1 + z_t * cand_t

        if not self.drop_candidates:
            if self.train:
                h_t = h_t * mask_t
            else:
                h_t = h_t * (1 - self.hid_dropout_rate)

        return h_t

    def _hidden_to_hidden_per_seq(self, xr_t, xz_t, xh_t, h_tm1, mask):
        return self._hidden_to_hidden(xr_t, xz_t, xh_t, mask, h_tm1)

    def set_phase(self, train):
        self.train = train

    def reset(self):
        self.h.set_value(np.zeros((self.batch_size, self.layer_size), dtype=theano.config.floatX))


class RNN:
    def __init__(self, input_size, layer_size, batch_size, hid_dropout_rate, drop_candidates, per_step,
                 activation=T.tanh,
                 weight_init=Uniform()):
        self.srng = RandomStreams(seed=np.random.randint(1000000))
        self.per_step = per_step

        W_shape = input_size, layer_size
        U_shape = layer_size, layer_size
        b_shape = layer_size

        self.W = theano.shared(weight_init(W_shape), name="W_rnn")
        self.U = theano.shared(weight_init(U_shape), name="U_rnn")
        self.b = theano.shared(weight_init(b_shape), name="b_rnn")
        self.h = theano.shared(np.zeros((batch_size, layer_size), dtype=theano.config.floatX))

        self.params = [
            self.W, self.U, self.b,
        ]

        self.activation = activation

        self.updates = []
        self.shared_state = False
        self.layer_size = layer_size
        self.batch_size = batch_size
        self.hid_dropout_rate = hid_dropout_rate
        self.train = True

    def __call__(self, x):
        # (time_steps, batch_size, layers_size)
        x = self._input_to_hidden(x)

        if self.per_step:
            masks = self.srng.binomial(n=1, p=1-self.hid_dropout_rate, size=(x.shape[0], self.batch_size, self.layer_size), dtype=theano.config.floatX)
            outputs, updates = theano.scan(self._hidden_to_hidden,
                sequences=[x, masks],
                outputs_info=[self.h],
                non_sequences=[self.U],
            )
        else:
            mask = self.srng.binomial(n=1, p=1-self.hid_dropout_rate, size=(self.batch_size, self.layer_size), dtype=theano.config.floatX)
            outputs, updates = theano.scan(self._hidden_to_hidden_per_seq,
                sequences=[x],
                outputs_info=[self.h],
                non_sequences=[self.U, mask],
            )

        self.updates = [(self.h, outputs[-1])]

        return outputs.dimshuffle((1, 0, 2))

    def _input_to_hidden(self, x):
        # (time_steps, batch_size, input_size)
        x = x.dimshuffle((1, 0, 2))

        x_t = T.dot(x, self.W) + self.b
        return x_t

    def _hidden_to_hidden(self, x_t, mask_t, h_tm1, u):
        if self.train:
            h_tm1 = h_tm1 * mask_t
        else:
            h_tm1 *= (1 - self.hid_dropout_rate)
        h_t = self.activation(x_t + T.dot(h_tm1, u))
        return h_t

    def _hidden_to_hidden_per_seq(self, x_t, h_tm1, u, mask):
        return self._hidden_to_hidden(x_t, mask, h_tm1, u)

    def set_phase(self, train):
        self.train = train

    def reset(self):
        self.h.set_value(np.zeros((self.batch_size, self.layer_size), dtype=theano.config.floatX))


class LastStepPooling(object):

    def __init__(self):
        pass

    def __call__(self, x):
        return x[:, -1, :]
