#!/usr/bin/env python
import optparse, sys, os, logging
from collections import defaultdict
import copy, math

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

e_word_count = defaultdict(int)
f_word_count = defaultdict(int)
fe_word_count = defaultdict(int)
# e_group_count = defaultdict(int)
# f_group_count = defaultdict(int)
for (n, (f, e)) in enumerate(bitext):
  for f_i in set(f):
    f_word_count[f_i] += 1
    for e_j in set(e):
      fe_word_count[(f_i,e_j)] += 1
      # e_group_count[e_j] += 1
      # f_group_count[f_i] += 1
  for e_j in set(e):
    e_word_count[e_j] += 1


# sum_e = sum(e_group_count.values())
# sum_f = sum(f_group_count.values())
sum_e = sum(e_word_count.values())
sum_f = sum(f_word_count.values())
sum_fe = sum(fe_word_count.values())
# print sum_e, sum_f, sum_fe
# for word in f_group_count:
#     print word
#     print f_group_count[word]

N = opts.num_sents
# t_k = defaultdict(float)
V_f = len(f_word_count)
t_k = dict.fromkeys(fe_word_count.keys(), 1.0/V_f)

def probXY(f_word, e_word):
    count_fe = 0
    for (k, (f_i, e_j)) in enumerate(fe_word_count.keys()):
        if f_i == f_word and e_j == e_word:
            count_fe += fe_word_count[(f_i, e_j)]
    return 1.0 * count_fe / sum_fe

def probbarXY(f_word, e_word):
    count_fbar_e, count_f_ebar, count_fbar_ebar = 0, 0, 0
    for (k, (f_i, e_j)) in enumerate(fe_word_count.keys()):
        if f_i != f_word and e_j == e_word:
            count_fbar_e += fe_word_count[(f_i, e_j)]
        elif f_i == f_word and e_j != e_word:
            count_f_ebar += fe_word_count[(f_i, e_j)]
        elif f_i != f_word and e_j != e_word:
            count_fbar_ebar += fe_word_count[(f_i, e_j)]
    return 1.0 * count_fbar_e / sum_fe, 1.0 * count_f_ebar / sum_fe, 1.0 * count_fbar_ebar / sum_fe

print probbarXY("transports", ",")
# def probXYbar(f_word, e_word):
#     count_fe = 0
#     for (k, (f_i, e_j)) in enumerate(fe_word_count.keys()):
#         if f_i == f_word and e_j != e_word:
#             count_fe += fe_word_count[(f_i, e_j)]
#     return 1.0 * count_fe / sum_fe


# def probXbarYbar(f_word, e_word):
#     count_fe = 0
#     for (k, (f_i, e_j)) in enumerate(fe_word_count.keys()):
#         if f_i != f_word and e_j != e_word:
#             count_fe += fe_word_count[(f_i, e_j)]
#     return 1.0 * count_fe / sum_fe


sum_e_LLR_score = defaultdict(float)
for (k, (f_i, e_j)) in enumerate(fe_word_count.keys()):
    # p_f = 1.0 * f_group_count[f_i] / sum_f
    # p_e = 1.0 * e_group_count[e_j] / sum_e
    p_f = 1.0 * f_word_count[f_i] / sum_f
    p_e = 1.0 * e_word_count[e_j] / sum_e
    p_fe = 1.0 * fe_word_count[(f_i, e_j)] / sum_fe
    # print f_i, p_f, p_e, p_fe
    # print f_i, f_group_count[f_i], sum_f


    if p_fe > p_f * p_e:
        # p_f_e_bar = 1.0 * (f_group_count[f_i] - fe_word_count[(f_i, e_j)]) / sum_fe
        # p_f_bar_e = 1.0 * (e_group_count[e_j] - fe_word_count[(f_i, e_j)]) / sum_fe
        # p_f_bar_e_bar = 1.0 * (sum_fe - fe_word_count[(f_i, e_j)]) / sum_fe
        # p_f_e_bar = probXYbar(f_i, e_j)
        # p_f_bar_e = probXbarY(f_i, e_j)
        # p_f_bar_e_bar = probXbarYbar(f_i, e_j)
        p_f_e_bar, p_f_bar_e, p_f_bar_e_bar = probbarXY(f_i, e_j)

        f_e = p_fe * math.log10(p_fe / (p_f * p_e))
        f_e_bar = p_f_e_bar * math.log10(p_f_e_bar / (p_f * (1 - p_e)))
        if p_f_bar_e / ((1 - p_f) * p_e) <=0:
            print p_f_bar_e / ((1 - p_f) * p_e)

        f_bar_e = p_f_bar_e * math.log10(p_f_bar_e / ((1 - p_f) * p_e))
        f_bar_e_bar = p_f_bar_e_bar * math.log10(p_f_bar_e_bar / ((1 - p_f) * (1 - p_e)))
        # print f_e_bar
        # print "\n"


        t_k[(f_i, e_j)] = 2 * N * (f_e + f_e_bar + f_bar_e + f_bar_e_bar)
        sum_e_LLR_score[e_j] += t_k[(f_i, e_j)]
        # print t_k[(f_i, e_j)]

max_sum_e_LLR_score = max(sum_e_LLR_score.values())

# e_sum = defaultdict(float)
for f,e in t_k:
    t_k[(f,e)] = t_k[(f,e)] / max_sum_e_LLR_score
    # print t_k[(f,e)]
    # e_sum[e] += t_k[(f,e)]

# print t_k[("transports", ",")]
# print max(t_k.values())
# print min(t_k.values())
# print max(e_sum.values())
# print min(e_sum.values())



# V_f = len(f_word_count)
# t_k = dict.fromkeys(fe_word_count.keys(), 1.0/V_f)

V = V_f
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
    t_k[(f_i, e_j)] = 1.0 * fe_count[(f_i,e_j)] / e_count[e_j]
    # t_k[(f_i, e_j)] = (1.0 * fe_count[(f_i,e_j)] + n) / (e_count[e_j] + n * V)
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


del e_word_count
del f_word_count
del fe_word_count
# del e_group_count
# del f_group_count
del t_k
del sum_e_LLR_score
del fe_count
del e_count
