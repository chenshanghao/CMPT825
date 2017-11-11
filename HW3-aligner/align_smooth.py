#!/usr/bin/env python
import optparse, sys, os, logging
from collections import defaultdict
import copy

optparser = optparse.OptionParser()
optparser.add_option("-d", "--datadir", dest="datadir", default="data", help="data directory (default=data)")
optparser.add_option("-p", "--prefix", dest="fileprefix", default="hansards", help="prefix of parallel data files (default=hansards)")
optparser.add_option("-e", "--english", dest="english", default="en", help="suffix of English (target language) filename (default=en)")
optparser.add_option("-f", "--french", dest="french", default="fr", help="suffix of French (source language) filename (default=fr)")
optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="filename for logging output")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="threshold for alignment (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.french)
e_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.english)

if opts.logfile:
    logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.INFO)

sys.stderr.write("Training IBM Model 1 (no nulls) with Expectation Maximization...\n")
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]

# for i in xrange(len(bitext)):
#   e_sentence = bitext[i][1]
#   bitext[i][1] = ["NullWord"] * (len(e_sentence)/10) + e_sentence

f_word_count = defaultdict(int)
fe_word_count = defaultdict(int)
for (n, (f, e)) in enumerate(bitext):
  for f_i in set(f):
    f_word_count[f_i] += 1
    for e_j in set(e):
      fe_word_count[(f_i,e_j)] += 1

V_f = len(f_word_count)
t_k = dict.fromkeys(fe_word_count.keys(), 1.0/V_f)

V = 1000
n = 1
# Precision = 0.294292
# Recall = 0.399208
# AER = 0.669803


iteration = 5

for i in xrange(iteration):
  sys.stderr.write("Iteration {0}\n".format(i))
  fe_count = defaultdict(int)
  e_count = defaultdict(int)
  for (n, (f, e)) in enumerate(bitext):
    for f_i in f:
      Z = 0
      for e_j in e:
        Z += t_k[(f_i,e_j)]
      for e_j in e:
        c = 1.0 * t_k[(f_i,e_j)] / Z
        fe_count[(f_i,e_j)] += c
        e_count[e_j] += c

  for (k, (f_i, e_j)) in enumerate(fe_count.keys()):
    # t_k[(f_i, e_j)] = 1.0 * fe_count[(f_i,e_j)] / e_count[e_j]
    t_k[(f_i, e_j)] = (1.0 * fe_count[(f_i,e_j)] + n) / (e_count[e_j] + n * V)
    if k % 50000 == 0:
      sys.stderr.write(".")
  sys.stderr.write("\n")

sys.stderr.write("Aligning...\n")
for (n, (f, e)) in enumerate(bitext):
  for (i, f_i) in enumerate(f):
    bestp = 0.0
    bestj = 0.0
    for (j, e_j) in enumerate(e):
      if t_k[(f_i, e_j)] > bestp:
        bestp = t_k[(f_i, e_j)]
        bestj = j
    sys.stdout.write("%i-%i " % (i, bestj))
  sys.stdout.write("\n")
