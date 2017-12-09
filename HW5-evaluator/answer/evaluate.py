#!/usr/bin/env python
# pylint: disable = I0011, E0401, C0103, C0321
'''evaluate module to automatically score the nlp tools'''
import argparse # optparse is deprecated
from itertools import islice # slicing for iterators
import string
import nltk
import nltk.corpus
import sklearn.ensemble
import numpy as np

alpha, beta, gamma, delta = .1, 3, .1, 1.
max_n = 4.
count = lambda s1, s2: sum(1. for word in s1 if word in s2)
list_sub = lambda f1, f2: list(a-b for a, b in zip(f1, f2))
feature_len = 33

def word_matches(hypo, ref):
    '''lambda count decorator
    get word count in both h and e
    '''
    return count(hypo, ref)

def add_delta_smoothing(numerator, denominator, cnt):
    '''add 1 smoothing'''
    return (numerator+delta)/(denominator+cnt*delta)

def ngram_acc(n, hypo, ref):
    '''calculate the ngram level precision and recall
    prec: ngrams in hypo can match any ref
    recall: match ngram in each ref separately, and then pick min err??
    '''
    if n > len(hypo) or n > len(ref): return 0.01, 0.01, 0.01
    unigram_hypo = hypo
    unigram_ref = ref
    denom_ref, denom_hypo = [], []
    for tmp in [unigram_hypo[i:i+n] for i in xrange(len(unigram_hypo)-n+1)]:
        denom_hypo += [' '.join(tmp)]
    for tmp in (unigram_ref[i:i+n] for i in xrange(len(unigram_ref)-n+1)):
        denom_ref += [' '.join(tmp)]
    matched = word_matches(denom_hypo, denom_ref)
    precision = add_delta_smoothing(matched, len(denom_hypo), len(denom_ref))
    recall = add_delta_smoothing(matched, len(denom_ref), len(denom_ref))
    fmean = precision*recall/((1-alpha)*recall+alpha*precision)
    return precision, recall, fmean

def word_count(hypo, ref):
    '''measures the len diff between hypo and ref. THIS sentense itself is ambigous!!!'''
    # normalize result by dividing len(ref)
    return 1.*(len(hypo)-len(ref))/len(ref)

def function_content_word_count(hypo, ref):
    '''grammatical word count, stopword in nltk
    return: stopword_count, contentword_count
    '''
    func_hypo = (lambda w: w in nltk.corpus.stopwords.words('english'), hypo)
    func_ref = (lambda w: w in nltk.corpus.stopwords.words('english'), ref)
    # return 1.*abs(len(content_hypo)-len(content_ref))/len(ref),\
    #        1.*abs(len(hypo)-len(content_hypo)-len(ref)+len(content_ref))/len(ref)
    return 1.*(len(func_hypo)-len(func_ref))/len(ref),\
           1.*(len(hypo)-len(func_hypo)-len(ref)+len(func_ref))/len(ref)

def punctuation_count(hypo, ref):
    '''calculate punctuation marks counts'''
    punc_hypo = count(hypo, set(string.punctuation))
    punc_ref = count(ref, set(string.punctuation))
    # return 1.*abs(punc_hypo-punc_ref)/len(ref)
    return 1.*(punc_hypo-punc_ref)/len(ref)

def features(hypo, ref):
    '''generate the 33 features for input hypothesis'''
    POS_hypo = [pair[1] for pair in nltk.pos_tag(hypo)]
    POS_ref = [pair[1] for pair in nltk.pos_tag(ref)]
    # s_ref, s_POS_ref = ' '.join(ref), ' '.join(POS_ref)
    #0,    1,    2,    3,    4,    5,    6,    7,    8,    9,    10,   11
    #1g_p, 2g_p, 3g_p, 4g_p, 1g_r, 2g_r, 3g_r, 4g_r, 1g_f, 2g_f, 3g_f, 4g_f
    #12,       13,       14,           15,       16,       17,       18,       19,       20
    #avg_prec, word_cnt, stopword_cnt, punc_cnt, cont_cnt, 1g_POS_p, 2g_POS_p, 3g_POS_p, 4g_POS_p
    #21,       22,       23,       24,       25,       26,       27,       28
    #1g_POS_r, 2g_POS_r, 3g_POS_r, 4g_POS_r, 1g_POS_f, 2g_POS_f, 3g_POS_f, 4g_POS_f
    #29,       30,       31,       32
    #1g_mix_p, 2g_mix_p, 3g_mix_p, 4g_mix_p
    feats = [0.] * feature_len
    feats[0], feats[4], feats[8] = ngram_acc(1, hypo, ref)
    feats[1], feats[5], feats[9] = ngram_acc(2, hypo, ref)
    feats[2], feats[6], feats[10] = ngram_acc(3, hypo, ref)
    feats[3], feats[7], feats[11] = ngram_acc(4, hypo, ref)
    feats[12] = sum(feats[0:int(max_n)])/max_n
    feats[13] = word_count(hypo, ref)
    feats[14], feats[16] = function_content_word_count(hypo, ref)
    feats[15] = punctuation_count(hypo, ref)
    feats[17], feats[21], feats[25] = ngram_acc(1, POS_hypo, POS_ref)
    feats[18], feats[22], feats[26] = ngram_acc(2, POS_hypo, POS_ref)
    feats[19], feats[23], feats[27] = ngram_acc(3, POS_hypo, POS_ref)
    feats[20], feats[24], feats[28] = ngram_acc(4, POS_hypo, POS_ref)
    feats[29], _, _ = ngram_acc(1, hypo+POS_hypo, ref+POS_ref)
    feats[30], _, _ = ngram_acc(2, hypo+POS_hypo, ref+POS_ref)
    feats[31], _, _ = ngram_acc(3, hypo+POS_hypo, ref+POS_ref)
    feats[32], _, _ = ngram_acc(4, hypo+POS_hypo, ref+POS_ref)

    return feats

def main():
    '''main entry'''
    parser = argparse.ArgumentParser(description='Evaluate translation hypotheses.')
    parser.add_argument('-i', '--input', default='data/hyp1-hyp2-ref',
                        help='input file (default data/hyp1-hyp2-ref)')
    parser.add_argument('-n', '--num_sentences', default=None, type=int,
                        help='Number of hypothesis pairs to evaluate')
    # note that if x == [1, 2, 3], then x[:None] == x[:] == x (copy); no need for sys.maxint
    opts = parser.parse_args()
    x_train_path = 'data/train-test.hyp1-hyp2-ref'
    y_train_path = 'data/train.gold'
    def sentences(filepath):
        '''we create a generator and avoid loading all sentences into a list
        modified to generate tokens instead of words
        '''
        with open(filepath) as f:
            for pair in f:
                pair = pair.decode('utf-8').encode('ascii', 'ignore').lower()
                yield [nltk.word_tokenize(sentence.strip()) for sentence in pair.split(' ||| ')]

    x_train = []
    if opts.num_sentences:
        y_train = np.loadtxt(y_train_path, delimiter='\n', usecols=0)[:opts.num_sentences]
    else:
        y_train = np.loadtxt(y_train_path, delimiter='\n', usecols=0)
    y_train = y_train.reshape(len(y_train), )
    y_train = -1*y_train
    for h1, h2, ref in islice(sentences(x_train_path), len(y_train)):
        f1 = features(h1, ref)
        f2 = features(h2, ref)
        x_train.append(list_sub(f1, f2))
    x_train = np.array(x_train)
    # clf = sklearn.ensemble.RandomForestClassifier(max_depth=100, random_state=0)
    clf = sklearn.ensemble.GradientBoostingClassifier(max_depth=5)
    clf.fit(x_train, y_train)

    for h1, h2, ref in sentences(opts.input):
        f1 = features(h1, ref)
        f2 = features(h2, ref)
        x = np.array(list_sub(f1, f2)).reshape(1, -1)
        print int(clf.predict(x)[0])


# convention to allow import of this file as a module
if __name__ == '__main__':
    main()
