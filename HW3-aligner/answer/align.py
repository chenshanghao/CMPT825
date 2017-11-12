'''
this code will align words between different languages
'''
#!/usr/bin/env python
#pylint: disable = I0011, E0401, C0103, C0321, W1401, C0301
from collections import defaultdict
import logging
import optparse
import os
import sys

PARSER = optparse.OptionParser()
PARSER.add_option("-d", "--datadir", dest="datadir", default="data", help="data directory (default=data)")
PARSER.add_option("-p", "--prefix", dest="fileprefix", default="hansards", help="prefix of parallel data files (default=hansards)")
PARSER.add_option("-e", "--english", dest="english", default="en", help="suffix of English (target language) filename (default=en)")
PARSER.add_option("-f", "--french", dest="french", default="fr", help="suffix of French (source language) filename (default=fr)")
PARSER.add_option("-l", "--logfile", dest="logfile", default=None, help="filename for logging output")
PARSER.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="threshold for alignment (default=0.5)")
PARSER.add_option("-n", "--num_sentences", dest="num_sents", default=10, type="int", help="Number of sentences to use for training and alignment")
(OPTS, _) = PARSER.parse_args()
f_data = "%s.%s" % (os.path.join(OPTS.datadir, OPTS.fileprefix), OPTS.french)
e_data = "%s.%s" % (os.path.join(OPTS.datadir, OPTS.fileprefix), OPTS.english)

if OPTS.logfile:
    logging.basicConfig(filename=OPTS.logfile, filemode='w', level=logging.INFO)

bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:OPTS.num_sents]]
EPOCH = 5

def init():
    '''new init method'''
    pass


def train_ef(params_ef, default):
    '''train e|f'''
    for k in xrange(EPOCH):
        sys.stderr.write("train_ef: Iteration {0}.\n".format(k))
        f_count = {}
        fe_count = {}
        for (f, e) in bitext:
            for e_j in set(e):
                Z = sum(params_ef.get((f_i, e_j), 1./default) for f_i in set(f))
                for f_i in set(f):
                    c = params_ef.get((f_i, e_j), 1./default)/Z
                    fe_count[(f_i, e_j)] = fe_count.get((f_i, e_j), 0.0)+c
                    f_count[f_i] = f_count.get(f_i, 0.0)+c
        #loss = 0.0
        for f_i, e_j in fe_count:
            #loss = max(abs(dice_ef.get((f_i, e_j), 1./default)-fe_count[(f_i, e_j)] / f_count[f_i]), loss)
            params_ef[(f_i, e_j)] = fe_count[(f_i, e_j)] / f_count[f_i]
        #sys.stderr.write(": loss is {0:2.5f}.\n".format(loss))

def train_fe(params_fe, default):
    '''train f|e'''
    for k in xrange(EPOCH):
        sys.stderr.write("train_fe: Iteration {0}.\n".format(k))
        e_count, fe_count = {}, {}
        for (f, e) in bitext:
            for f_i in set(f):
                Z = sum(params_fe.get((f_i, e_j), 1./default) for e_j in set(e))
                for e_j in set(e):
                    c = params_fe.get((f_i, e_j), 1./default)/Z
                    fe_count[(f_i, e_j)] = fe_count.get((f_i, e_j), 0.0)+c
                    e_count[e_j] = e_count.get(e_j, 0.0)+c
        #loss = 0.0
        for f_i, e_j in fe_count:
            #loss = max(abs(dice_fe.get((f_i, e_j), 1./default)-fe_count[(f_i, e_j)] / e_count[e_j]), loss)
            params_fe[(f_i, e_j)] = fe_count[(f_i, e_j)] / e_count[e_j]
        #sys.stderr.write(": loss is {0:2.5f}.\n".format(loss))


def train_symmetric(params, params_ef, params_fe):
    '''train posterior'''
    for k in xrange(EPOCH):
        sys.stderr.write("train_ef: Iteration {0}".format(k))
        e_count = defaultdict(float)
        f_count = defaultdict(float)
        fe_count = defaultdict(float)
        ef_count = defaultdict(float)
        for (f, e) in bitext:
            x = ((f_i, e_j)for e_j in e for f_i in f)
            for (f_i, e_j) in x:
                Z1 = sum(params_fe[(f_i, e_j)] for e_j in e)
                Z2 = sum(params_ef[(f_i, e_j)] for f_i in f)
                for e_prime in e:
                    fe_count[(f_i, e_prime)] += params_fe[(f_i, e_j)]/Z1
                    e_count[e_prime] += params_fe[(f_i, e_j)]/Z1
                for f_prime in f:
                    ef_count[(f_prime, e_j)] += params_ef[(f_i, e_j)]/Z2
                    f_count[f_prime] += params_ef[(f_i, e_j)]/Z2

        loss = 0.0
        for f_i, e_j in fe_count:
            loss = max(abs(params_ef[(f_i, e_j)]-fe_count[(f_i, e_j)] / f_count[f_i]), loss)
            params[(f_i, e_j)] = max(fe_count[(f_i, e_j)] / f_count[f_i] if f_i in f_count else 0,
                                   ef_count[(f_i, e_j)] / e_count[e_j] if e_j in e_count else 0)
        sys.stderr.write("; loss is {0:2.5f}.\n".format(loss))



def decode1(params_ef, params_fe):
    '''decode best alignment based on EM'''
    for f, e in bitext:
        res_ef = set()
        res_fe = set()
        for j, e_j in enumerate(e):
            bestp, besti = 0.0, 0.0
            for i, f_i in enumerate(f):
                if params_ef[(f_i, e_j)] > bestp:
                    bestp = params_ef[(f_i, e_j)]
                    besti = i
            res_ef.add((besti, j))
        for i, f_i in enumerate(f):
            bestp, bestj = 0.0, 0.0
            for j, e_j in enumerate(e):
                if params_fe[(f_i, e_j)] > bestp:
                    bestp = params_fe[(f_i, e_j)]
                    bestj = j
            res_fe.add((i, bestj))
        for _i, _j in res_ef&res_fe:
            sys.stdout.write("%i-%i " % (_i, _j))
        sys.stdout.write("\n")

def decode2(dice):
    '''decode best alignment based on EM'''
    for f, e in bitext:
        for i, f_i in enumerate(f):
            bestp, bestj = 0.0, 0.0
            for j, e_j in enumerate(e):
                if dice[(f_i, e_j)] > bestp:
                    bestp = dice[(f_i, e_j)]
                    bestj = j
            sys.stdout.write("%i-%i " % (i, bestj))
        sys.stdout.write("\n")


def decode3(params_ef, params_fe):
    '''decode best alignment based on EM'''
    for f, e in bitext:
        for i, f_i in enumerate(f):
            bestp, bestj = 0.0, 0.0
            for j, e_j in enumerate(e):
                if (f_i, e_j) in params_ef and (f_i, e_j) in params_fe:
                    pr = params_ef[(f_i, e_j)]*params_fe[(f_i, e_j)]
                    if pr > bestp:
                        bestp = pr
                        bestj = j
            sys.stdout.write("%i-%i " % (i, bestj))
        sys.stdout.write("\n")

def main():
    '''main'''
    sys.stderr.write("Training with expected maximization algorithm...\n")
    e_vocab = len(set([w for s in open(e_data) for w in s.strip().split()]))
    f_vocab = len(set([w for s in open(f_data) for w in s.strip().split()]))
    params_ef = {}
    params_fe = {}
    train_ef(params_ef, e_vocab)
    train_fe(params_fe, f_vocab)
    decode1(params_ef, params_fe)


if __name__ == "__main__":
    main()
