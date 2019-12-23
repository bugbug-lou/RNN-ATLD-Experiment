import pickle
import argparse

from database import LMDatabase
from sgd_optimizer import SGDOptimizer
import config

from layers import *
from model import Model


def main(args):
    x = T.imatrix()
    t = T.imatrix()

    data_dir = 'data'

    if config.model == 'rnn':
        opt_name = '%s.drop_%.02f.per_step_%d' % (config.model, args.hid_dropout_rate, args.per_step)
    else:
        opt_name = '%s.drop_%.02f.drop_cand_%d.per_step_%d' % (config.model, args.hid_dropout_rate, args.drop_candidates, args.per_step)

    sample_size = 15 if config.model == 'rnn' else 35
    overlap = 5 if config.model == 'rnn' else -1

    train_db = LMDatabase(data_dir, "train", sample_size=sample_size, overlap_size=overlap, batch_size=config.batch_size)
    valid_db = LMDatabase(data_dir, "valid", sample_size=sample_size, batch_size=train_db.batch_size)

    rnns = {
        'lstm': LSTM,
        'gru': GRU,
        'rnn': RNN
    }

    model = Model(x, t, sample_size, [
        Embed(train_db.vocab_size, config.layer_size, weight_init=Uniform(config.scale)),
        Dropout(config.in_dropout_rate),
        rnns[config.model](config.layer_size, config.layer_size, train_db.batch_size, args.hid_dropout_rate,
                           args.hid_scale, args.drop_candidates, args.per_step, weight_init=Uniform(config.scale)),
        Dropout(config.out_dropout_rate),
        Linear(config.layer_size, train_db.vocab_size, weight_init=Uniform(config.scale))
    ])

    clip = {
        'lstm': 10,
        'gru': 20,
        'rnn': 30
    }

    opt = SGDOptimizer(model, x, t, train_db, test_db=valid_db,
                       name=opt_name,
                       clip_gradients=True, clip_threshold=clip[config.model],
                       print_norms=True)

    lr = {
        'lstm': 1,
        'gru': 0.1,
        'rnn': 0.05
    }

    if not args.test_only:
        opt.train(train_db, test_db=valid_db, learning_rate=lr[config.model], lr_decay=1.5, epochs=100)

    model.load("exp/%s/opt.pkl" % opt_name)
    test_datasets = pickle.load(open("%s/test.pkl" % data_dir))
    for d in test_datasets:
        valid_db.dataset = test_datasets[d]
        valid_db.bounds, ins, outs = valid_db.create_dataset()
        valid_db.ins.set_value(ins)
        valid_db.outs.set_value(outs)
        costs = opt.get_test_costs(valid_db)
        print d, costs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hid_dropout_rate', dest='hid_dropout_rate', type=float, default=0.0)
    parser.add_argument('--test', dest='test_only', action='store_true', default=False)
    parser.add_argument('--hid_scale', dest='hid_scale', type=float, default=0.0)
    parser.add_argument('--drop_candidates', dest='drop_candidates', action='store_true', default=False)
    parser.add_argument('--per_step', dest='per_step', action='store_true', default=False)
    args, _ = parser.parse_known_args()

    main(args)
