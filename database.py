import pickle
import theano
import numpy


class LMDatabase:
    def __init__(self, data_dir, phase, batch_size=100, sample_size=10, overlap_size=-1):
        self.batch_size = batch_size
        self.phase = phase
        self.dataset = pickle.load(open("%s/%s.pkl" % (data_dir, phase)))
        self.vocab_size = max(self.dataset) + 1
        print "%s: %d words" % (phase, len(self.dataset))
        self.sample_size = sample_size
        self.overlap_size = overlap_size if overlap_size > 0 else sample_size
        self.bounds, ins, outs = self.create_dataset()
        self.batch_number = len(self.bounds)
        self.ins = theano.shared(ins)
        self.outs = theano.shared(outs)
        self.current_batch = 0
        self.new_epoch = False

    def new_epoch_started(self):
        if self.new_epoch:
            self.new_epoch = False
            return True
        return False

    def create_dataset(self):
        n_overlap = self.sample_size / self.overlap_size
        assert n_overlap >= 1
        x_out_list = []
        target_list = []
        for o in xrange(n_overlap):
            x_in = numpy.asarray(self.dataset)
            x_in = x_in[o * self.overlap_size: x_in.shape[0] - self.sample_size + o * self.overlap_size]
            if x_in.ndim != 1:
                raise ValueError("Data must be 1D, was", x_in.ndim)

            if x_in.shape[0] % (self.batch_size*self.sample_size) == 0:
                print(" x_in.shape[0] % (batch_size*model_seq_len) == 0 -> x_in is "
                      "set to x_in = x_in[:-1]")
                x_in = x_in[:-1]

            x_resize =  \
                (x_in.shape[0] // (self.batch_size*self.sample_size))*self.sample_size*self.batch_size
            n_samples = x_resize // (self.sample_size)
            n_batches = n_samples // self.batch_size

            targets = x_in[1:x_resize+1].reshape(n_samples, self.sample_size)
            x_out = x_in[:x_resize].reshape(n_samples, self.sample_size)

            out = numpy.zeros(n_samples, dtype=int)
            b = []
            for i in range(n_batches):
                val = range(i, n_batches*self.batch_size+i, n_batches)
                out[i*self.batch_size:(i+1)*self.batch_size] = val
                b.append((i * self.batch_size * n_overlap, (i+1) * self.batch_size * n_overlap))

            x_out = x_out[out]
            #x_out = x_out.reshape((x_out.shape[0], x_out.shape[1], 1))
            targets = targets[out]
            x_out_list.append(x_out)
            target_list.append(targets)

        x_out = numpy.zeros((x_out_list[0].shape[0] * n_overlap, x_out_list[0].shape[1]))
        targets = numpy.zeros((target_list[0].shape[0] * n_overlap, target_list[0].shape[1]))
        for i in xrange(x_out.shape[0]):
            idx1 = i % n_overlap
            idx2 = i / n_overlap
            x_out[i, :] = x_out_list[idx1][idx2]
            targets[i, :] = target_list[idx1][idx2]
        self.batch_size *= n_overlap
        return b, x_out.astype('int32'), targets.astype('int32')

    def total_batches(self):
        return self.batch_number

    def get_bounds(self, update=True):
        b = self.bounds[self.current_batch]
        if update:
            self.current_batch += 1
        if self.current_batch >= self.batch_number:
            self.current_batch = 0
            self.new_epoch = True
        return b