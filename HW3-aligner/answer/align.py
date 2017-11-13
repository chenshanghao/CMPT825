#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals

import optparse, sys, os, logging
from itertools import chain
from collections import defaultdict

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

sys.stderr.write("Training with IBM's coefficient...\n")
bitext = [[f.strip().split(), e.strip().split() + [None]] for (f, e) in zip(open(f_data), open(e_data))[:opts.num_sents]]
vecb_f = chain.from_iterable((i for i in set(f)) for f, _ in iter(bitext))
vecb_e = chain.from_iterable((i for i in set(e)) for _, e in iter(bitext))
v_f = len(set(vecb_f))
v_e = len(set(vecb_e))

fe_t = defaultdict(lambda: 1./v_f)
ef_t = defaultdict(lambda: 1./v_e)



# train p(f|e)
for _ in range(5):
    sys.stderr.write("Iteration {} with P(f|e)\n".format(_+1))
    e_count = defaultdict(float)
    fe_count = defaultdict(float)

    # variables for estimating a
    # count_a = defaultdict(float)
    # total_a = defaultdict(float)
    # s_total = defaultdict(float)

    for (n, (f, e)) in enumerate(bitext):
        l_e, l_f = len(e), len(f)
        for f_i in set(f):
            z = sum((fe_t[(f_i, e_j)] for e_j in set(e)))
            for e_j in set(e):
                c = fe_t[(f_i, e_j)]/z
                fe_count[(f_i, e_j)] += c
                e_count[e_j] += c
        if n % 500 == 0:
            sys.stderr.write(".")
    sys.stderr.write('\n')

    for (k, (f_i, e_j)) in enumerate(fe_count.iterkeys()):
        fe_t[(f_i, e_j)] = fe_count[(f_i, e_j)]/e_count[e_j]
    #del e_count
    #del fe_count

# train p(e|f)
for _ in range(5):
    sys.stderr.write("Iteration {} with P(e|f)\n".format(_+1))
    f_count = defaultdict(float)
    ef_count = defaultdict(float)
    for (n, (f, e)) in enumerate(bitext):
        for e_i in set(e):
            z = sum((ef_t[(e_i, f_j)] for f_j in set(f)))
            for f_j in set(f):
                c = ef_t[(e_i, f_j)]/z
                ef_count[(e_i, f_j)] += c
                f_count[f_j] += c
        if n % 500 == 0:
            sys.stderr.write(".")
    sys.stderr.write('\n')

    for (k, (e_i, f_j)) in enumerate(ef_count.iterkeys()):
        ef_t[(e_i, f_j)] = ef_count[(e_i, f_j)]/f_count[f_j]
    #del f_count
    #del ef_count

# add smoothing  temporary
n = 0.01
V = 10000

for (f, e) in fe_count:
    fe_t[f, e] = (fe_count[(f, e)] + n)/ (e_count[e] + n*V)
for (e, f) in ef_count:
    ef_t[e, f] = (ef_count[(e, f)] + n) / (f_count[f] + n*V)

del(fe_count)
del(ef_count)
del(e_count)
del(f_count)

def grow_diag(e, f, f2e, e2f):
    # this method can improve recall which lower precision
    neighboring = ((-1,0),(0,-1),(1,0),(0,1),(-1,-1),(-1,1),(1,-1),(1,1))
    alignment = e2f & f2e
    union_alignment = e2f | f2e
    keep_loop = True
    while keep_loop:
        keep_loop = False
        new_alignment = []
        aligned_i, aligned_j = zip(*alignment)
        for (i, f_i) in enumerate(f):
            for (j, e_j) in enumerate(e):
                if (i, j) in alignment:
                    new_alignment = [
                        (new_i, new_j) for (new_i, new_j) in
                        [(i+d_i, j+d_j) for (d_i, d_j) in neighboring]
                        if (new_i not in aligned_i or new_j not in aligned_j) and (new_i, new_j) in union_alignment
                    ]
                if new_alignment:
                    keep_loop = True
                    alignment.update(new_alignment)
    return alignment

# intersection decoding
sys.stderr.write('Decoding...\n')
for (f, e) in bitext:
    align_set_fe = set()
    align_set_ef = set()
    for (i, f_i) in enumerate(f):
        bestp = 0
        bestj = 0
        for (j, e_j) in enumerate(e):
            if fe_t[(f_i, e_j)] > bestp:
                bestp = fe_t[(f_i, e_j)]
                bestj = j
        align_set_fe.add((i, bestj))
    for (j, e_j) in enumerate(e):
        bestp = 0
        besti = 0
        for (i, f_i) in enumerate(f):
            if ef_t[(e_j, f_i)] > bestp:
                bestp = ef_t[(e_j, f_i)]
                besti = i
        align_set_ef.add((besti, j))
    for (i, j) in grow_diag(e, f, align_set_fe, align_set_ef):
        if e[j] is not None:
            sys.stdout.write("%i-%i " % (i, j))
    sys.stdout.write("\n")