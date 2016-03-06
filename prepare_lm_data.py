from alphabet import Alphabet
import utils

import pickle
import numpy

data = "data"
dataset = "ptb"


def load_dataset(filename, alphabet):
    with open(filename) as f:
        lines = f.readlines()

    for l in lines:
        l = l.split(" ")
        if dataset == "ptb":
            l = l[1:]
        for w in l:
            alphabet.add(w)


def make_alphabet():
    alphabet = Alphabet(0)
    load_dataset("%s/%s/%s.train.txt" % (data, dataset, dataset), alphabet)
    load_dataset("%s/%s/%s.valid.txt" % (data, dataset, dataset), alphabet)
    if dataset == 'ptb':
        load_dataset("%s/%s/%s.test.txt" % (data, dataset, dataset), alphabet)

    print "%s: total %d words" % (dataset, len(alphabet))

    pickle.dump(alphabet, open("%s/alphabet.pkl" % data, "w"))


def parse_dataset(phase):
    dataset_fname = "%s/%s/%s.%s.txt" % (data, dataset, dataset, phase)

    alphabet = pickle.load(open("%s/alphabet.pkl" % data))

    with open(dataset_fname) as f:
        lines = f.readlines()

    words = []
    for l in lines:
        l = l.split(" ")
        if dataset == "ptb":
            l = l[1:]
        words += [alphabet[w] for w in l]

    pickle.dump(words, open("%s/%s.pkl" % (data, phase), 'w'))

    print phase, "done"


def parse_test(test_datasets):
    if dataset == 'ptb':
        phases = ['valid', 'test']
    elif dataset == 'text8':
        phases = ['valid']
    else:
        raise Exception("unknown dataset %s" % dataset)

    alphabet = pickle.load(open("%s/alphabet.pkl" % data))

    for p in phases:
        fname = "%s/%s/%s.%s.txt" % (data, dataset, dataset, p)
        with open(fname) as f:
            lines = f.readlines()
        words = []
        for l in lines:
            l = l.split(" ")
            if dataset == "ptb":
                l = l[1:]
            words.extend([alphabet[w] for w in l])

        test_datasets[p] = words

        print fname, "done"


if __name__ == "__main__":
    make_alphabet()

    parse_dataset("train")
    parse_dataset("valid")

    test_datasets = {}
    parse_test(test_datasets)
    pickle.dump(test_datasets, open("%s/test.pkl" % data, 'w'))
