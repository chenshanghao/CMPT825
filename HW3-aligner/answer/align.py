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

def init():
    '''uniformly initilize fe_count'''
    f_word_len = len(set([w for s in open(f_data) for w in s.strip().split()]))
    return 1.0/f_word_len


def train():
    '''train body, dice is t_k'''
    dice_default = init()
    sys.stderr.write("Training with Dice's coefficient...\n")
    for k in range(5):
        sys.stderr.write("Iteration {0} ".format(k))
        #f_count = {}
        e_count = {}
        fe_count = {}

        for (f, e) in bitext:
            for f_i in f:
                Z = 0.0
                for e_j in e:
                    if (f_i, e_j) not in dice: dice[(f_i, e_j)]=dice_default
                    Z += dice[(f_i, e_j)]
                for e_j in e:
                    c = dice[(f_i, e_j)]/Z
                    fe_count[(f_i, e_j)] = fe_count.get((f_i, e_j), 0.0)+c
                    e_count[e_j] = e_count.get(e_j, 0.0)+c
        for (it, (f_i, e_j)) in enumerate(fe_count.keys()):
            dice[(f_i, e_j)] = 1.0*fe_count[(f_i, e_j)]/e_count[e_j]
            if it % 50000 == 0:
                sys.stderr.write(".")
        sys.stderr.write("\n")


def decode():
    for (f, e) in bitext:
        for i, f_i in enumerate(f):
            bestp, bestj = 0.0, 0.0
            for j, e_j in enumerate(e):
                if dice[(f_i, e_j)] > bestp:
                    bestp = dice[(f_i, e_j)]
                    bestj = j
            sys.stdout.write("%i-%i " % (i, bestj))
        sys.stdout.write("\n")

def main():
    '''main'''
    train()
    decode()

if __name__ == "__main__":
    main()
