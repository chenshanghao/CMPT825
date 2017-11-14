'''
this code will align words between different languages
'''
#!/usr/bin/env python
#pylint: disable = I0011, E0401, C0103, C0321, W1401, C0301
import collections
import math
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
N = OPTS.num_sents
smooth_n = 0.01

class _keydefaultdict(collections.defaultdict):
    '''define a local function for uniform probability initialization'''
    def __init__(self, default_factory=None, *a, **kw):
        '''initialization'''
        if (default_factory is not None and
                not isinstance(default_factory, collections.Callable)):
            raise TypeError('first argument must be callable')
        self.default_factory = default_factory

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret

def init():
    '''TODO: too much mem, horrible 8.8G isn't enough new init method'''
    e_word = [w for i, s in enumerate(open(e_data)) if i < N for w in s.strip().split()]
    f_word = [w for i, s in enumerate(open(f_data)) if i < N for w in s.strip().split()]
    num_e = len(e_word)
    num_f = len(f_word)
    e_word_count = dict(collections.Counter(e_word))
    f_word_count = dict(collections.Counter(f_word))
    fe_word_count = collections.defaultdict(int)

    # total occurrence number of single f word and single e word
    total_f = sum(f_word_count.values())
    total_e = sum(e_word_count.values())
    for (f, e) in bitext:
        for f_i in f:
            for e_j in e:
                fe_word_count[(f_i, e_j)] += 1
    total_fe = sum(fe_word_count.values())

    # total occurrence number of (f, e) respect to a specific f word and all other e words
    margin_f = {}
    # total occurrence number of (f, e) respect to a specific e word and all other f words
    margin_e = {}
    for f_i in f_word:
        margin_f[f_i] = sum(fe_word_count[(f_i, e_j)] for e_j in e_word)
    for e_j in xrange(num_e):
        margin_e[e_j] = sum(fe_word_count[(f_i, e_j)] for f_i in f_word)
    del e_word, f_word

    t_k = collections.defaultdict(float)
    sum_e_LLR_score = collections.defaultdict(float)

    for f_i, e_j in fe_word_count.keys():
        C_f_e = fe_word_count[(f_i, e_j)]
        p_f = 1.0 * f_word_count[f_i] / total_f
        p_e = 1.0 * e_word_count[e_j] / total_e
        P_fe = 1.0 * C_f_e / total_fe
        if P_fe > p_f * p_e:
            C_f_Note = margin_f[f_i] - C_f_e
            C_Notf_e = margin_e[e_j] - C_f_e
            C_Notf_Note = total_fe - C_f_Note - C_Notf_e + C_f_e
            if C_f_Note == 0 or C_Notf_e == 0:
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
                t_k[(f_i, e_j)] = 2 * N * (G_f_e + G_f_Note + G_Notf_e + G_Notf_Note)
        else:
            t_k[(f_i, e_j)] = 1.0 / num_f

        sum_e_LLR_score[e_j] += t_k[(f_i, e_j)]

    max_sum_e_LLR_score = max(sum_e_LLR_score.values())
    for f, e in t_k:
        t_k[(f, e)] = t_k[(f, e)] / max_sum_e_LLR_score

    return t_k


def train_ef_ibm1(params_ef, default):
    '''train e|f'''
    for k in xrange(EPOCH):
        sys.stderr.write("train_ef: Iteration {0}.\n".format(k))
        f_count, fe_count = {}, {}
        for (f, e) in bitext:
            for e_j in set(e):
                Z = sum(params_ef.get((f_i, e_j), 1./default) for f_i in set(f))
                for f_i in set(f):
                    c = params_ef.get((f_i, e_j), 1./default)/Z
                    fe_count[(f_i, e_j)] = fe_count.get((f_i, e_j), 0.0)+c
                    f_count[f_i] = f_count.get(f_i, 0.0)+c
        for f_i, e_j in fe_count:
            params_ef[(f_i, e_j)] = (fe_count[(f_i, e_j)]+smooth_n) / (f_count[f_i]+smooth_n*N)

def train_fe_ibm1(params_fe, default):
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
        for f_i, e_j in fe_count:
            params_fe[(f_i, e_j)] = (fe_count[(f_i, e_j)]+smooth_n) / (e_count[e_j]+smooth_n*N)


def key_fun(key):
    ''' default_factory function for keycollections.defaultdict '''
    i, j, len_f, len_e = key
    return math.exp(-abs(1.*(len_e*i-len_f*j)/(len_e*len_f)))

def train_fe_ibm2(params_fe, default, align):
    '''add alignment params'''
    Z_align = sum(align.values())
    for k in xrange(EPOCH):
        sys.stderr.write("train_fe: Iteration {0}.\n".format(k))
        e_count, fe_count = {}, {}
        for (f, e) in bitext:
            for i, f_i in enumerate(f):
                Z = sum(params_fe.get((f_i, e_j), align[(i, j, len(f), len(e))]/(default*Z_align)) for j, e_j in enumerate(e))
                for j, e_j in enumerate(e):
                    c = params_fe.get((f_i, e_j), align[(i, j, len(f), len(e))]/(default*Z_align))/Z
                    fe_count[(f_i, e_j)] = fe_count.get((f_i, e_j), 0.0)+c
                    e_count[e_j] = e_count.get(e_j, 0.0)+c
        for f_i, e_j in fe_count:
            params_fe[(f_i, e_j)] = (fe_count[(f_i, e_j)]+smooth_n) / (e_count[e_j]+smooth_n*N)

def train_ef_ibm2(params_ef, default, align):
    '''add alignment params'''
    Z_align = sum(align.values())
    for k in xrange(EPOCH):
        sys.stderr.write("train_ef: Iteration {0}.\n".format(k))
        f_count, fe_count = {}, {}
        for (f, e) in bitext:
            for j, e_j in enumerate(e):
                Z = sum(params_ef.get((f_i, e_j), align[(j, i, len(e), len(f))]/(default*Z_align)) for i, f_i in enumerate(f))
                for i, f_i in enumerate(f):
                    c = params_ef.get((f_i, e_j), align[(j, i, len(e), len(f))]/(default*Z_align))/Z
                    fe_count[(f_i, e_j)] = fe_count.get((f_i, e_j), 0.0)+c
                    f_count[f_i] = f_count.get(f_i, 0.0)+c
        for f_i, e_j in fe_count:
            params_ef[(f_i, e_j)] = (fe_count[(f_i, e_j)]+smooth_n) / (f_count[f_i]+smooth_n*N)

def grow_diag(e, f, f2e, e2f):
    '''this method can improve recall which lower precision'''
    neighboring = ((-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    alignment = e2f & f2e
    union_alignment = e2f | f2e
    keep_loop = True
    while keep_loop:
        keep_loop = False
        new_alignment = []
        aligned_i, aligned_j = zip(*alignment)
        for i in xrange(len(f)):
            for j in xrange(len(e)):
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

def decode1(align_ef, params_ef, align_fe, params_fe):
    '''decode best alignment based on EM'''
    Z_align_ef = sum(align_ef.values())
    Z_align_fe = sum(align_fe.values())
    for f, e in bitext:
        res_ef = set()
        res_fe = set()
        for j, e_j in enumerate(e):
            bestp, besti = 0.0, 0.0
            for i, f_i in enumerate(f):
                pr = params_ef[(f_i, e_j)]*align_ef[(j, i, len(e), len(f))]/Z_align_ef
                if pr > bestp:
                    bestp = pr
                    besti = i
            res_ef.add((besti, j))
        for i, f_i in enumerate(f):
            bestp, bestj = 0.0, 0.0
            for j, e_j in enumerate(e):
                pr = params_fe[(f_i, e_j)]*align_fe[(i, j, len(f), len(e))]/Z_align_fe
                if pr > bestp:
                    bestp = pr
                    bestj = j
            res_fe.add((i, bestj))
        for (i, j) in grow_diag(e, f, res_fe, res_ef):
        #for (i, j) in res_fe&res_ef:
            if e[j] is not None:
                sys.stdout.write("%i-%i " % (i, j))
        sys.stdout.write("\n")

def decode2(params_fe):
    '''decode best alignment based on EM'''
    for f, e in bitext:
        for i, f_i in enumerate(f):
            bestp, bestj = 0.0, 0.0
            for j, e_j in enumerate(e):
                if params_fe[(f_i, e_j)] > bestp:
                    bestp = params_fe[(f_i, e_j)]
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
    train_ef_ibm1(params_ef, e_vocab)
    train_fe_ibm1(params_fe, f_vocab)

    align_fe = _keydefaultdict(key_fun)
    for f, e in bitext:
        for i in xrange(len(f)):
            for j in xrange(len(e)):
                align_fe[(i, j, len(f), len(e))]
    train_fe_ibm2(params_fe, f_vocab, align_fe)

    align_ef = _keydefaultdict(key_fun)
    for f, e in bitext:
        for j in xrange(len(e)):
            for i in xrange(len(f)):
                align_ef[(j, i, len(e), len(f))]
    train_ef_ibm2(params_ef, e_vocab, align_ef)
    decode1(align_ef, params_ef, align_fe, params_fe)


if __name__ == "__main__":
    main()
