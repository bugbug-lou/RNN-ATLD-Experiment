import argparse

from synth_database import SyntheticDatabase
from temp_order import TempOrder
from sgd_optimizer import SGDOptimizer
from model import Model
from layers import *

import config


class TempOrderModel(Model):

    def __init__(self, input, target, layers):
        super(TempOrderModel, self).__init__(input, target, 0, layers)

    @property
    def costs(self):
        p = self.output(self.input)
        cost = CrossEntropy()(p, self.target)

        pred = T.argmax(p, axis=1)
        true = T.argmax(self.target, axis=1)
        acc = T.mean(T.eq(pred, true))

        return [cost, acc]


def main(args):
    x = T.tensor3()
    t = T.matrix()

    task = TempOrder(args.low, args.high, args.length, batch_size=config.batch_size, long_sequences=False)
    train_db = SyntheticDatabase(task, number_of_batches=config.number_of_batches)
    valid_db = SyntheticDatabase(task, number_of_batches=config.test_sequences/config.batch_size, phase='valid')

    model = TempOrderModel(x, t, [
        LSTM(task.input_size, config.layer_size, config.batch_size, args.hid_dropout_rate,
             args.drop_candidates, args.per_step,
             weight_init=Uniform(config.scale), persistent=False),
        LastStepPooling(),
        Linear(config.layer_size, task.output_size)
    ])

    if args.finetune:
        model.load('exp/temp_order/opt.pkl')

    #  debug
    st = T.iscalar()
    end = T.iscalar()
    out = model.output(model.input)
    out = model.costs
    f = theano.function([st, end], out,
                        givens={
                            x: train_db.ins[st:end],
                            t: train_db.outs[st:end]
                        },
                        on_unused_input='ignore')
    b = train_db.get_bounds()
    out = f(b[0], b[1])
    #  debug

    opt = SGDOptimizer(model, x, t, train_db, test_db=valid_db, name="temp_order",
                       clip_gradients=True, clip_threshold=5, print_norms=True)

    opt.train(train_db, test_db=valid_db, learning_rate=args.lr, epochs=1000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hid_dropout_rate', dest='hid_dropout_rate', type=float, default=0.0)
    parser.add_argument('--drop_candidates', dest='drop_candidates', action='store_true', default=False)
    parser.add_argument('--per_step', dest='per_step', action='store_true', default=False)
    parser.add_argument('--low', default=10, type=int)
    parser.add_argument('--high', default=20, type=int)
    parser.add_argument('--length', default=30, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--finetune', action='store_true', default=False)
    args, _ = parser.parse_known_args()

    print args

    main(args)
