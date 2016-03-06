import theano
import numpy
import random


class SyntheticDatabase:
    def __init__(self, task, number_of_batches=2000, update=False, phase='train'):
        self.task = task
        self.batch_number = number_of_batches
        self.current_batch = 0
        self.update = update
        self.phase = phase
        self.new_epoch = False

        ins, outs, self.bounds = self.create_dataset()
        self.ins = theano.shared(ins)
        self.outs = theano.shared(outs)

    def create_dataset(self):
        bounds = []
        ins = []
        outs = []
        curr_idx = 0
        for i in xrange(self.batch_number):
            curr_ins, curr_outs = self.task.get_batch()
            assert curr_ins.shape[0] > 0
            bounds.append((curr_idx, curr_idx+curr_ins.shape[0]))
            curr_idx += curr_ins.shape[0]
            ins.append(curr_ins)
            outs.append(curr_outs)

        ins = numpy.concatenate(ins, axis=0)
        outs = numpy.concatenate(outs, axis=0)

        ins = ins.astype(theano.config.floatX)
        outs = outs.astype(theano.config.floatX)

        return ins, outs, bounds

    def total_batches(self):
        return self.batch_number

    def get_bounds(self):
        if self.current_batch >= self.batch_number:
            if self.update:
                ins, outs, self.bounds = self.create_dataset()
                self.ins.set_value(ins)
                self.outs.set_value(outs)
                print "updated"
            self.current_batch = 0
            random.shuffle(self.bounds)
            self.new_epoch = True

        b = self.bounds[self.current_batch]
        self.current_batch += 1
        return b

    def new_epoch_started(self):
        if self.new_epoch:
            self.new_epoch = False
            return True
        return False
