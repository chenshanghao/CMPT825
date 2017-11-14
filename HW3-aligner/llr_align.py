#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
for (n, (f, e)) in enumerate(bitext):
  for f_i in set(f):
    f_word_count[f_i] += 1
    for e_j in set(e):
      fe_word_count[(f_i,e_j)] += 1
  for e_j in set(e):
    e_word_count[e_j] += 1

# print fe_word_count[("MATIÈRES", "contents")]
# print fe_word_count[("CRÉDITS", "supply")]
# print fe_word_count[("prière", "prayers")]
# print fe_word_count[("prière", "prayers")]


# f words and e words list
f_list = f_word_count.keys()
e_list = e_word_count.keys()

# total occurrence number of single f word and single e word
total_f = sum(f_word_count.values())
total_e = sum(e_word_count.values())
total_fe = sum(fe_word_count.values())

# number of f words and e words
num_f = len(f_list)
num_e = len(e_list)

# a matrix to repesent the combinations of f word and e word
fe_matrix = [[0 for x in range(num_e)] for y in range(num_f)]

# total occurrence number of (f, e) respect to a specific f word and all other e words
sum_f = [0] * num_f
# total occurrence number of (f, e) respect to a specific e word and all other f words
sum_e = [0] * num_e

for i in range(num_f):
  for j in range(num_e):
    fe_matrix[i][j] = fe_word_count[(f_list[i], e_list[j])]
    sum_f[i] += fe_matrix[i][j]

for j in range(num_e):
  for i in range(num_f):
    sum_e[j] += fe_matrix[i][j]

t_k = defaultdict(float)
sum_e_LLR_score = defaultdict(float)
N = opts.num_sents

for (k, (f_i, e_j)) in enumerate(fe_word_count.keys()):
  f_index = f_list.index(f_i)
  e_index = e_list.index((e_j))

  C_f_e = fe_matrix[f_index][e_index]

  p_f = 1.0 * f_word_count[f_i] / total_f
  p_e = 1.0 * e_word_count[e_j] / total_e
  P_fe = 1.0 * C_f_e / total_fe

  if P_fe > p_f * p_e:
    C_f_Note = sum_f[f_index] - C_f_e
    C_Notf_e = sum_e[e_index] - C_f_e
    C_Notf_Note = total_fe - C_f_Note - C_Notf_e + C_f_e

    if C_f_Note == 0 or C_Notf_e == 0: 
    # i.e. sum_f[f_index] == C_f_e or sum_e[e_index] == C_f_e
    # this indicates that f_i is only related to e_j in the original dataset
      t_k[(f_i, e_j)] = 1
    else:
      P_f_Note = 1.0 * C_f_Note / total_fe
      P_Notf_e = 1.0 * C_Notf_e / total_fe
      P_Notf_Note = 1.0 * C_Notf_Note / total_fe

      G_f_e = P_fe * math.log10(P_fe / (p_f * p_e))

      Div_f_Note = P_f_Note / (p_f * (1 - p_e))
      if Div_f_Note > 0:
        G_f_Note = P_f_Note * math.log10(Div_f_Note)

      Div_Notf_e = P_Notf_e / ((1 - p_f) * p_e)
      if Div_Notf_e > 0:
        G_Notf_e = P_Notf_e * math.log10(Div_Notf_e)

      Div_Notf_Note = P_Notf_Note / ((1 - p_f) * (1 - p_e))
      if Div_Notf_Note > 0:
        G_Notf_Note = P_Notf_Note * math.log10(Div_Notf_Note)

      '''
      if P_f_Note / (p_f * (1 - p_e)) <=0:
        print "P_f_Note:"
        print f_index, e_index, C_f_e, fe_word_count[(f_i, e_j)]
        print C_f_Note, sum_f[f_index], C_f_e
        print P_f_Note
        print f_i, e_j, P_f_Note / (p_f * (1 - p_e))
        print "\n"

      if P_Notf_e / ((1 - p_f) * p_e) <=0:
        print "P_Notf_e:"
        print f_index, e_index, C_f_e, fe_word_count[(f_i, e_j)]
        print C_Notf_e, sum_f[f_index], C_f_e
        print P_Notf_e
        print f_i, e_j, P_Notf_e / ((1 - p_f) * p_e)

      if P_Notf_Note / ((1 - p_f) * (1 - p_e)) <=0:
        print "P_Notf_Note:"
        print f_index, e_index, C_f_e, fe_word_count[(f_i, e_j)]
        print C_Notf_Note, sum_f[f_index], C_f_e
        print P_Notf_Note
        print f_i, e_j, P_Notf_Note / ((1 - p_f) * (1 - p_e))
      '''

      t_k[(f_i, e_j)] = 2 * N * (G_f_e + G_f_Note + G_Notf_e + G_Notf_Note)
  else:
    t_k[(f_i, e_j)] = 1.0 / num_f

  sum_e_LLR_score[e_j] += t_k[(f_i, e_j)]

max_sum_e_LLR_score = max(sum_e_LLR_score.values())

for f,e in t_k:
  t_k[(f,e)] = t_k[(f,e)] / max_sum_e_LLR_score

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
del t_k
del sum_e_LLR_score
del fe_count
del e_count
