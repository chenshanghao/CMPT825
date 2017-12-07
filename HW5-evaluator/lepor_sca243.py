#!/usr/bin/env python
import argparse # optparse is deprecated
from itertools import islice # slicing for iterators
import sys
from nltk.stem import *    #.porter
from nltk.corpus import wordnet as wn
from nltk import pos_tag
import math
import numpy
 

count = lambda s1, s2: sum(1. for word in s1 if word in s2)
alpha, delta =.1, 1.

def exact_word_matches(h, ref):
    bitvec_ref = [False for _ in ref]
    bitvec_h = [False for _ in h]
    count = 0
    for index, word in enumerate(h):
        for ref_index, ref_word in enumerate(ref):
            if not bitvec_ref[ref_index] and word == ref_word:
                count += 1
                bitvec_ref[ref_index] = True
                bitvec_h[index] = True
                break
    return (count, bitvec_h, bitvec_ref)

def porter_word_matches(h, ref, bitvec_h, bitvec_ref):
    count = 0
    for i, word in enumerate(h):
        if not bitvec_h[i]:
            for j, ref_word in enumerate(ref):
                if not bitvec_ref[j]:
                    if h[i] == ref[j]:
                        count += 1
                        bitvec_ref[j] = True
                        bitvec_h[i] = True
                        break
    return (count, bitvec_h, bitvec_ref)

def stemming(arr):
    stemmer = PorterStemmer()
    stem_arr = [stemmer.stem(word.decode('utf-8')) for word in arr]
    return stem_arr

def synonyms(arr):
    syn_arr = []
    for word in arr:
        syn_arr.append([lst for ss in wn.synsets(word.decode('utf-8')) for lst in ss.lemma_names()])  
    return syn_arr

def wordnet_word_matches(h, ref, bitvec_h, bitvec_ref):
    count = 0
    for i, syns in enumerate(h):
        if not bitvec_h[i]:
            for j, ref_syn in enumerate(ref):
                if not bitvec_ref[j]:
                    if check_syns(syns, ref_syn):
                        count += 1
                        bitvec_ref[j] = True
                        bitvec_h[i] = True
                        break
    return (count, bitvec_h, bitvec_ref)

# Have you sinned?
def check_syns(h, ref):
    for synset in h:
        if synset in ref:
            return True
    return False

def calc_chunks(h, ref, h_stem, ref_stem, h_syn, ref_syn):
    chunk = 0
    bitvec = [False for _ in ref]
    i = 0
    j = 0
    changed = False
    while i < len(h):
        j = 0
        changed = False
        while j < len(ref):
            if i < len(h) and j < len(ref) and (h[i] == ref[j] or h_stem[i] == ref_stem[j] or check_syns(h_syn[i], ref_syn[j])) and not bitvec[j]:
                while i < len(h) and j < len(ref) and (h[i] == ref[j] or h_stem[i] == ref_stem[j] or check_syns(h_syn[i], ref_syn[j])) and not bitvec[j]:
                    bitvec[j] = True
                    changed = True
                    i += 1
                    j += 1
                chunk += 1
            else:
                j += 1
        i = i + 1 if not changed else i
    return chunk

def add_delta_smoothing(numerator, denominator, cnt):
    '''add 1 smoothing'''
    return (numerator+delta)/(denominator+cnt*delta)


def word_matches(hypo, ref):
    '''lambda count decorator
    get word count in both h and e
    '''
    return count(hypo, ref)

def ngram_acc(n, hypo, ref):
    '''calculate the ngram level precision and recall
    prec: ngrams in hypo can match any ref
    recall: match ngram in each ref separately, and then pick min err??
    '''
    if n > len(hypo) or n > len(ref): return 0.01, 0.01, 0.01
    unigram_hypo = hypo
    unigram_ref = ref
    # s_ref = ' '.join(ref)
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
 
def main():
    parser = argparse.ArgumentParser(description='Evaluate translation hypotheses.')
    parser.add_argument('-i', '--input', default='data/hyp1-hyp2-ref', help='input file (default data/hyp1-hyp2-ref)')
    parser.add_argument('-n', '--num_sentences', default=None, type=int, help='Number of hypothesis pairs to evaluate')
    parser.add_argument('-a', '--alpha', default=0.82, type=float, help='Balances precision and recall')
    parser.add_argument('-b', '--beta', default=1.0, type=float, help='METEOR penalty parameter')
    parser.add_argument('-g', '--gamma', default=0.21, type=float, help='METEOR penalty parameter')
    # note that if x == [1, 2, 3], then x[:None] == x[:] == x (copy); no need for sys.maxint
    opts = parser.parse_args()
 
    # we create a generator and avoid loading all sentences into a list
    def sentences():
        with open(opts.input) as f:
            for pair in f:
                yield [sentence.lower().strip().split() for sentence in pair.split(' ||| ')]
 
    # note: the -n option does not work in the original code
    for h1, h2, ref in islice(sentences(), opts.num_sentences):

        (h1_exact_matches, bitvec_h1, bitvec_ref1) = exact_word_matches(h1, ref)
        (h2_exact_matches, bitvec_h2, bitvec_ref2) = exact_word_matches(h2, ref)

        # porter stemmer:
        h1_stem = stemming(h1)
        ref_stem = stemming(ref)
        h2_stem = stemming(h2)


        (h1_porter_matches, bitvec_h1, bitvec_ref1) = porter_word_matches(h1_stem, ref_stem, bitvec_h1, bitvec_ref1)
        (h2_porter_matches, bitvec_h2, bitvec_ref2) = porter_word_matches(h2_stem, ref_stem, bitvec_h2, bitvec_ref2)

        # synsets wordnet:
        h1_syn = synonyms(h1)
        h2_syn = synonyms(h2)
        ref_syn = synonyms(ref)


        (h1_wn_matches, bitvec_h1, bitvec_ref1) = wordnet_word_matches(h1_syn, ref_syn, bitvec_h1, bitvec_ref1)
        (h2_wn_matches, bitvec_h2, bitvec_ref2) = wordnet_word_matches(h2_syn, ref_syn, bitvec_h2, bitvec_ref2)

        h1_num_matches = h1_exact_matches + h1_porter_matches + h1_wn_matches
        h2_num_matches = h2_exact_matches + h2_porter_matches + h2_wn_matches


        # precision and recall calculations:
        precision1 = h1_num_matches / float(len(h1))
        precision2 = h2_num_matches / float(len(h2))

        recall1 = h1_num_matches / float(len(ref))
        recall2 = h2_num_matches / float(len(ref))

        h1_score = (precision1 * recall1) / float(((1 - opts.alpha) * recall1) + (opts.alpha * precision1) + 1)
        h2_score = (precision2 * recall2) / float(((1 - opts.alpha) * recall2) + (opts.alpha * precision2) + 1)

        # meteor penalty:
        chunk1 = calc_chunks(h1, ref, h1_stem, ref_stem, h1_syn, ref_syn)
        chunk2 = calc_chunks(h2, ref, h2_stem, ref_stem, h2_syn, ref_syn)

        # sys.stderr.write("chunk1: %s\n" %(chunk1))

        frag1 = chunk1 / float(h1_num_matches) if h1_num_matches != 0 else 0
        frag2 = chunk2 / float(h2_num_matches) if h2_num_matches != 0 else 0

        penalty1 = opts.gamma * (frag1 ** opts.beta)
        penalty2 = opts.gamma * (frag2 ** opts.beta)

        h1_match = (1 - penalty1) * h1_score
        h2_match = (1 - penalty2) * h2_score


        # LEPOR:
        lp1 = math.exp(1 - (len(h1) / float(len(ref)))) if len(h1) > len(ref) else 1 if len(h1) == len(ref) else math.exp(1 - (len(ref) / float(len(h1))))
        lp2 = math.exp(1 - (len(h2) / float(len(ref)))) if len(h2) > len(ref) else 1 if len(h1) == len(ref) else math.exp(1 - (len(ref) / float(len(h2))))


        # n-gram:
        h1_precision_1, h1_recall_1, h1_femean_1 = ngram_acc(1, h1, ref)
        h1_precision_2, h1_recall_2, h1_femean_2 = ngram_acc(2, h1, ref)
        h1_precision_3, h1_recall_3, h1_femean_3 = ngram_acc(3, h1, ref)
        h1_precision_4, h1_recall_4, h1_femean_4 = ngram_acc(4, h1, ref)

        h2_precision_1, h2_recall_1, h2_femean_1 = ngram_acc(1, h2, ref)   
        h2_precision_2, h2_recall_2, h2_femean_2 = ngram_acc(2, h2, ref)
        h2_precision_3, h2_recall_3, h2_femean_3 = ngram_acc(3, h2, ref)
        h2_precision_4, h2_recall_4, h2_femean_4 = ngram_acc(4, h2, ref)


        h1_match = (0.65 * h1_match * lp1) + (0.35 * (  h1_recall_1 + h1_recall_2 +  h1_recall_3))
        h2_match = (0.65 * h2_match * lp2) + (0.35 * (  h2_recall_1 + h2_recall_2 +  h1_recall_3))

        print(1 if h1_match > h2_match else # \begin{cases}
                (0 if h1_match == h2_match
                    else -1)) # \end{cases}

 
# convention to allow import of this file as a module
if __name__ == '__main__':
    main()