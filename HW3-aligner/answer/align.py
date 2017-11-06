'''
this code will align words between different languages
'''
#!/usr/bin/env python
#pylint: disable = I0011, E0401, C0103, C0321, W1401, C0301
import logging
import optparse
import os
import sys

OPTPARSER = optparse.OptionParser()
OPTPARSER.add_option("-d", "--datadir", dest="datadir", default="data", help="data directory (default=data)")
OPTPARSER.add_option("-p", "--prefix", dest="fileprefix", default="hansards", help="prefix of parallel data files (default=hansards)")
OPTPARSER.add_option("-e", "--english", dest="english", default="en", help="suffix of English (target language) filename (default=en)")
OPTPARSER.add_option("-f", "--french", dest="french", default="fr", help="suffix of French (source language) filename (default=fr)")
OPTPARSER.add_option("-l", "--logfile", dest="logfile", default=None, help="filename for logging output")
OPTPARSER.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="threshold for alignment (default=0.5)")
OPTPARSER.add_option("-n", "--num_sentences", dest="num_sents", default=1000, type="int", help="Number of sentences to use for training and alignment")
(OPTS, _) = OPTPARSER.parse_args()
f_data = "%s.%s" % (os.path.join(OPTS.datadir, OPTS.fileprefix), OPTS.french)
e_data = "%s.%s" % (os.path.join(OPTS.datadir, OPTS.fileprefix), OPTS.english)

if OPTS.logfile:
    logging.basicConfig(filename=OPTS.logfile, filemode='w', level=logging.INFO)

bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:OPTS.num_sents]]
dice = {}
dice_ef = {}
dice_fe = {}

def train_ef():
    '''train e|f, dice is t_k'''
    def init():
        '''uniformly initilize ef_count'''
        e_word_len = len(set([w for s in open(e_data) for w in s.strip().split()]))
        return 1.0/e_word_len

    dice_default = init()
    for k in range(5):
        sys.stderr.write("train_ef: Iteration {0} ".format(k))
        f_count = {}
        fe_count = {}

        for (f, e) in bitext:
            for e_j in e:
                Z = 0.0
                for f_i in f:
                    if (f_i, e_j) not in dice_ef: dice_ef[(f_i, e_j)] = dice_default
                    Z += dice_ef[(f_i, e_j)]
                for f_i in f:
                    c = dice_ef[(f_i, e_j)]/Z
                    fe_count[(f_i, e_j)] = fe_count.get((f_i, e_j), 0.0)+c
                    f_count[f_i] = f_count.get(f_i, 0.0)+c
        loss = 0.0
        for (it, (f_i, e_j)) in enumerate(fe_count.keys()):
            loss = max(abs(dice_ef[(f_i, e_j)]-fe_count[(f_i, e_j)] / f_count[f_i]), loss)
            dice_ef[(f_i, e_j)] = fe_count[(f_i, e_j)] / f_count[f_i]
            if it % 500000 == 0:
                sys.stderr.write(".")
        sys.stderr.write("; loss is {0:2.5f}.\n".format(loss))
        if k == 5: return


def train_fe():
    '''train f|e, dice is t_k'''
    def init():
        '''uniformly initilize fe_count'''
        f_word_len = len(set([w for s in open(f_data) for w in s.strip().split()]))
        return 1.0/f_word_len

    dice_default = init()
    for k in range(5):
        sys.stderr.write("train_fe: Iteration {0} ".format(k))
        e_count = {}
        fe_count = {}

        for (f, e) in bitext:
            for f_i in f:
                Z = 0.0
                for e_j in e:
                    if (f_i, e_j) not in dice_fe: dice_fe[(f_i, e_j)] = dice_default
                    Z += dice_fe[(f_i, e_j)]
                for e_j in e:
                    c = dice_fe[(f_i, e_j)]/Z
                    fe_count[(f_i, e_j)] = fe_count.get((f_i, e_j), 0.0)+c
                    e_count[e_j] = e_count.get(e_j, 0.0)+c
        loss = 0.0
        for (it, (f_i, e_j)) in enumerate(fe_count.keys()):
            loss = max(abs(dice_fe[(f_i, e_j)]-fe_count[(f_i, e_j)] / e_count[e_j]), loss)
            dice_fe[(f_i, e_j)] = fe_count[(f_i, e_j)] / e_count[e_j]
            if it % 500000 == 0:
                sys.stderr.write(".")
        sys.stderr.write("; loss is {0:2.5f}.\n".format(loss))
        #if loss < .0001: return
        if k == 5: return


def decode():
    '''decode best alignment based on EM'''
    for (f, e) in dice_ef:
        if (f, e) in dice_fe:
            dice[(f, e)] = dice_fe[(f, e)]
    for (f, e) in bitext:
        for i, f_i in enumerate(f):
            bestp, bestj = 0.0, 0.0
            for j, e_j in enumerate(e):
                if dice[(f_i, e_j)] > bestp:
                    bestp = dice[(f_i, e_j)]
                    bestj = j
            sys.stdout.write("%i-%i " % (i, bestj))
        sys.stdout.write("\n")

def train():
    '''train body'''
    sys.stderr.write("Training with expected maximization algorithm...\n")
    train_ef()
    train_fe()

def main():
    '''main'''
    train()
    decode()

if __name__ == "__main__":
    main()
