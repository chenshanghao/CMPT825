#!/usr/bin/env python
import argparse # optparse is deprecated
from itertools import islice # slicing for iterators
from math import pow
import re

def sentence_preprocess(word_list):
    lowercase_word_list = [word.lower() for word in word_list]
    dropThe_word_list = filter(lambda word: word != "the", lowercase_word_list)
    dropCharacter_word_list = map(lambda word: re.sub('[,.!?:;"]', '', word), dropThe_word_list)
    return dropCharacter_word_list


def word_matches(h, ref):
    return sum(1 for w in h if w in ref)

def meteor_calculate(h, ref):
    rset = set(ref)
    h_match = word_matches(h, rset)
    h_uni_precision = 1.0 * h_match / len(h)
    h_uni_recall = 1.0 * h_match / len(ref)
    harmonic_mean = 1.0 * 10 * h_uni_precision * h_uni_recall / (h_uni_recall + 9 * h_uni_precision)
    # h_chunk = 
    # penalty = 0.5 * math.pow(1.0 * h_chunk / h_match, 3)
    # meteor = harmonic_mean * (1 - penalty)
    meteor = harmonic_mean
    return meteor


def main():
    parser = argparse.ArgumentParser(description='Evaluate translation hypotheses.')
    # parser.add_argument('-i', '--input', default='data/hyp1-hyp2-ref',
    #         help='input file (default data/hyp1-hyp2-ref)')
    parser.add_argument('-i', '--input', default='data/small-hyp1-hyp2-ref',
            help='input file (default data/hyp1-hyp2-ref)')
    parser.add_argument('-n', '--num_sentences', default=None, type=int,
            help='Number of hypothesis pairs to evaluate')

    # note that if x == [1, 2, 3], then x[:None] == x[:] == x (copy); no need for sys.maxint
    opts = parser.parse_args()
 
    # we create a generator and avoid loading all sentences into a list
    def sentences():
        with open(opts.input) as f:
            for pair in f:
                yield [sentence.strip().split() for sentence in pair.split(' ||| ')]
 
    # note: the -n option does not work in the original code
    for h1, h2, ref in islice(sentences(), opts.num_sentences):
        h1 = sentence_preprocess(h1)
        h2 = sentence_preprocess(h2)
        ref = sentence_preprocess(ref)

        # h1_score = meteor_calculate(h1, ref)
        # h2_score = meteor_calculate(h2, ref)

        # print(1 if h1_score > h2_score else # \begin{cases}
        #         (0 if h1_score == h2_score
        #             else -1)) # \end{cases}
 
# convention to allow import of this file as a module
if __name__ == '__main__':
    main()
