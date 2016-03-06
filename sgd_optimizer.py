import theano
import theano.tensor as T
import numpy
import time
import pickle
import os
import shutil
from collections import OrderedDict


class SGDOptimizer:
    def __init__(self, model, inputs, targets, train_db,
                 clip_gradients=True, clip_threshold=1,
                 name=None, batch_size=0,
                 test_db=None,
                 print_norms=False):
        self.model = model
        self.batch_size = batch_size
        self.params = model.params
        self.print_norms = print_norms
        self.inputs = inputs
        self.costs, train_updates = model.build(train=True)
        self.targets = targets
        self.name = name
        self.best_pp = 1e6
        self.orig_costs_len = len(self.costs)

        self.start = T.iscalar()
        self.end = T.iscalar()

        self.log_f = None
        if name is not None:
            opt_folder = "exp/" + name
            if os.path.exists(opt_folder) is False:
                os.makedirs(opt_folder)

            log_filename = "%s/log.txt" % opt_folder
            print "log in %s" % log_filename
            if os.path.exists(log_filename):
                self.log_f = open(log_filename, "a", 0)
                print >>self.log_f, "-------------------"
            else:
                self.log_f = open(log_filename, "w", 0)
            if print_norms:
                self.norms_f = open("exp/" + name + "/norms.csv", "w", buffering=0)

        self.grads = [T.grad(self.costs[0], p) for p in self.params]

        self.grad_l2 = T.sqrt(sum([T.sum(g ** 2) for g in self.grads]))
        self.costs.append(self.grad_l2)

        self.lr = T.fscalar()
        self.mom = T.fscalar()

        updates = OrderedDict()
        for param, gr in zip(self.params, self.grads):
            if clip_gradients:
                upd = self.lr * T.switch(self.grad_l2 > clip_threshold, clip_threshold * gr / self.grad_l2, gr)
            else:
                upd = self.lr * gr
            updates[param] = param - upd

        for upd in train_updates:
            updates[upd[0]] = upd[1]

        self.train_net = theano.function([self.start, self.end, self.lr],
            self.costs,
            updates=updates,
            givens={
                self.inputs: train_db.ins[self.start:self.end],
                self.targets: train_db.outs[self.start:self.end]
            },
            allow_input_downcast=True,
            on_unused_input="warn"
        )

        self.test_function = None
        if test_db is not None:
            test_costs, test_updates = model.build(train=False)
            updates = OrderedDict()
            for upd in test_updates:
                updates[upd[0]] = upd[1]
            self.test_function = theano.function([self.start, self.end], test_costs,
                                        givens={
                                            self.inputs: test_db.ins[self.start:self.end],
                                            self.targets: test_db.outs[self.start:self.end]
                                        },
                                        updates=updates,
                                        on_unused_input='warn')

    def save_state(self, acc):
        if self.name is not None:
            opt_state_filename = "exp/%s/opt.pkl" % self.name
            prev_opt_state_filename = "exp/%s/prev_opt.pkl" % self.name
            if os.path.exists(opt_state_filename):
                shutil.move(opt_state_filename, prev_opt_state_filename)
            self.model.dump(opt_state_filename)

    def get_test_costs(self, test_db):
        self.model.reset()
        test_costs = []

        for i in xrange(test_db.total_batches()):
            b = test_db.get_bounds()
            c = self.test_function(b[0], b[1])
            c = c[0:self.orig_costs_len]
            test_costs.append(c)

        return numpy.average(test_costs, axis=0)

    def train(self, train_db, test_db=None, learning_rate=0.01, epochs=1000, lr_decay=1, output_frequency=10):
        cost = []
        t = time.time()
        iteration = 0
        total_iteration = 0
        epoch = 0
        while epoch < epochs:
            for i in xrange(train_db.total_batches()):
                iteration += 1
                total_iteration += 1

                if i == train_db.total_batches() / 2:
                    self.model.reset()

                b = train_db.get_bounds()
                net_outs = self.train_net(b[0], b[1], learning_rate)

                cost.append(net_outs[:self.orig_costs_len])

                if (epoch != 0) | (i != 0):
                    if self.print_norms:
                        print >>self.norms_f, "%d,%f" % (total_iteration, net_outs[-1])

                    if iteration % output_frequency == 0:
                        avg = numpy.mean(cost, axis=0)
                        complete = float(iteration) / train_db.total_batches()
                        print '\repoch %03i, %.04f done, cost = %s, took %s sec' % (epoch + 1, complete, str(avg), time.time() - t),
                        if self.log_f is not None:
                            print >>self.log_f, 'epoch %i, iter %i, cost=' % (epoch + 1, iteration), avg
                        cost = []

            if train_db.new_epoch_started():
                epoch += 1
                iteration = 0
                if test_db is not None:
                    print "testing...",
                    avg = self.get_test_costs(test_db)
                    self.save_state(avg[0])
                    print "validation cost = ", avg
                    if avg[1] < self.best_pp:
                        self.best_pp = avg[1]
                    else:
                        learning_rate /= lr_decay
                        if lr_decay != 1:
                            print "new lr = %f" % learning_rate
                    if self.log_f is not None:
                        print >>self.log_f, "validation cost =", avg
                self.model.reset()
                t = time.time()
