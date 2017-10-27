'''
this code will align words between different languages
'''
#!/usr/bin/env python
#pylint: disable = I0011, E0401, C0103, C0321, W1401, C0301
import logging
import optparse
import os
import sys
from collections import defaultdict

OPTPARSER = optparse.OptionParser()
OPTPARSER.add_option("-d", "--datadir", dest="datadir", default="data", help="data directory (default=data)")
OPTPARSER.add_option("-p", "--prefix", dest="fileprefix", default="hansards", help="prefix of parallel data files (default=hansards)")
OPTPARSER.add_option("-e", "--english", dest="english", default="en", help="suffix of English (target language) filename (default=en)")
OPTPARSER.add_option("-f", "--french", dest="french", default="fr", help="suffix of French (source language) filename (default=fr)")
OPTPARSER.add_option("-l", "--logfile", dest="logfile", default=None, help="filename for logging output")
OPTPARSER.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="threshold for alignment (default=0.5)")
OPTPARSER.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
(OPTS, _) = OPTPARSER.parse_args()
f_data = "%s.%s" % (os.path.join(OPTS.datadir, OPTS.fileprefix), OPTS.french)
e_data = "%s.%s" % (os.path.join(OPTS.datadir, OPTS.fileprefix), OPTS.english)

if OPTS.logfile:
    logging.basicConfig(filename=OPTS.logfile, filemode='w', level=logging.INFO)

sys.stderr.write("Training with Dice's coefficient...")
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:OPTS.num_sents]]
f_count = defaultdict(int)
e_count = defaultdict(int)
fe_count = defaultdict(int)
for (n, (f, e)) in enumerate(bitext):
    for f_i in set(f):
        f_count[f_i] += 1
    for e_j in set(e):
        fe_count[(f_i, e_j)] += 1
    for e_j in set(e):
        e_count[e_j] += 1
    if n % 500 == 0:
        sys.stderr.write(".")

dice = defaultdict(int)
for (k, (f_i, e_j)) in enumerate(fe_count.keys()):
    dice[(f_i, e_j)] = 2.0 * fe_count[(f_i, e_j)] / (f_count[f_i] + e_count[e_j])
    if k % 5000 == 0:
        sys.stderr.write(".")
sys.stderr.write("\n")

for (f, e) in bitext:
    for (i, f_i) in enumerate(f):
        for (j, e_j) in enumerate(e):
            if dice[(f_i, e_j)] >= OPTS.threshold:
                sys.stdout.write("%i-%i " % (i, j))
    sys.stdout.write("\n")
