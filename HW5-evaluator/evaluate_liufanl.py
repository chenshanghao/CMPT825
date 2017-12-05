#!/usr/bin/env python
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import argparse # optparse is deprecated
from itertools import islice # slicing for iterators
from math import pow, exp
import copy
# import nltk
# nltk.download('wordnet')
from nltk.stem.porter import *
from nltk.corpus import wordnet

ALPHA = 0.73
BETA = 1.0
GAMMA = 0.21

def sentence_preprocess(word_list):
    lowercase_word_list = [word.lower() for word in word_list]
    dropThe_word_list = filter(lambda word: word != "the", lowercase_word_list)
    dropCharacter_word_list = map(lambda word: word.strip(',.!?:;"$*%()[]<>/'), dropThe_word_list)
    return dropCharacter_word_list

def word_exact_matches(h, ref, h_matched):
    for i in xrange(len(h)):
        if h_matched[i] == False and h[i] in ref:
            h_matched[i] = True
            ref.remove(h[i])
    return h_matched, ref


def encode_unicode(word):
    if isinstance(word, unicode) == True:
        return word.encode('utf8')
    else:
        return word

def word_stemmed_matches(h, h_exact_matched, ref_not_exact_match):
    stemmer = PorterStemmer()
    stemmed_ref = map(lambda word: stemmer.stem(word), ref_not_exact_match)
    encode_ref = map(lambda word: encode_unicode(word) , stemmed_ref)
    ref_matched = [False] * len(ref_not_exact_match)

    for i in xrange(len(h)):
        if h_exact_matched[i] == False:
            stemmed_word = stemmer.stem(h[i])
            encode_word = encode_unicode(stemmed_word)
            if encode_word in encode_ref and ref_matched[encode_ref.index(encode_word)] == False:
                h_exact_matched[i] = True
                ref_matched[encode_ref.index(encode_word)] = True

    ref_not_stemmed_match = []
    for j in xrange(len(ref_not_exact_match)):
        if ref_matched[j] == False:
            ref_not_stemmed_match.append(ref_not_exact_match[j])

    return h_exact_matched, ref_not_stemmed_match

def word_synonym_matches(h, h_stemmed_matched, ref_not_stemmed_match):
    synonym_words = []
    for ref_word in ref_not_stemmed_match:
        ref_word_synonym = []
        for synset in wordnet.synsets(ref_word):
            for lemma in synset.lemmas():
                ref_word_synonym.append(lemma.name())
        synonym_words.append(list(set(ref_word_synonym)))

    for i in xrange(len(h)):
        if h_stemmed_matched[i] == False:
            for j in xrange(len(synonym_words)):
                if h[i] in synonym_words[j]:
                    h_stemmed_matched[i] = True
                    del synonym_words[j]
                    break
    return h_stemmed_matched

# def word_matches(h):
#     return sum(1 for w in h if w in ref)

def get_chunk(hypo, ref):
    '''TODO: need further test
    ref : the cat sat on the mat
    hypo: the cat was sat on the mat
    chunk = 2
    '''
    ref_sentence = ' '.join(ref)
    chunks = i = j = 0
    while i < len(hypo):
        if hypo[i] in ref:
            chunks += 1
            j = i
            hypo_shard = hypo[i]
            while hypo_shard in ref_sentence and j < len(hypo):
                j += 1
                hypo_shard = ' '.join(hypo[i:j+1])
        if j > i: i = j
        else: i += 1
    return chunks

def meteor_calculate(h, h_total_matched, ref):
    # rset = set(ref)
    # h_match = word_matches(h, rset)
    h_match = h_total_matched.count(True) + 0.1 # plus 0.1 to avoide zero match
    h_uni_precision = 1.0 * h_match / len(h_total_matched)
    h_uni_recall = 1.0 * h_match / len(ref)
    harmonic_mean = 1.0 * h_uni_precision * h_uni_recall / ( (1- ALPHA) * h_uni_recall + ALPHA * h_uni_precision)

    h_chunk = get_chunk(h, ref)
    chunk_penalty = GAMMA * pow(1.0 * h_chunk / h_match, BETA)
    meteor = harmonic_mean * (1 - chunk_penalty)

    return meteor

def length_penalty(h, ref):
    h_length = len(h)
    ref_length = len(ref)
    h_length_penalty = 0
    if h_length < ref_length:
        h_length_penalty = exp(1 - 1.0 * ref_length / h_length)
    elif h_length == ref_length:
        h_length_penalty = 1
    else: # h_length > ref_length
        h_length_penalty = exp(1 - 1.0 * h_length / ref_length)
    return h_length_penalty


def main():
    parser = argparse.ArgumentParser(description='Evaluate translation hypotheses.')
    parser.add_argument('-i', '--input', default='data/hyp1-hyp2-ref',
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

        h1_clean = sentence_preprocess(h1)
        h2_clean = sentence_preprocess(h2)
        ref_clean = sentence_preprocess(ref)
        ref_clean_h1 = copy.deepcopy(ref_clean)
        ref_clean_h2 = copy.deepcopy(ref_clean)

        h1_matched = [False] * len(h1_clean)
        h2_matched = [False] * len(h2_clean)

        # METEOR
        h1_exact_matched, ref_not_exact_match_h1 = word_exact_matches(h1_clean, ref_clean_h1, h1_matched)
        h2_exact_matched, ref_not_exact_match_h2 = word_exact_matches(h2_clean, ref_clean_h2, h2_matched)

        h1_stemmed_matched, ref_not_stemmed_match_h1 = word_stemmed_matches(h1_clean, h1_exact_matched, ref_not_exact_match_h1)
        h2_stemmed_matched, ref_not_stemmed_match_h2 = word_stemmed_matches(h2_clean, h2_exact_matched, ref_not_exact_match_h2)

        h1_synonym_matched = word_synonym_matches(h1_clean, h1_stemmed_matched, ref_not_stemmed_match_h1)
        h2_synonym_matched = word_synonym_matches(h2_clean, h2_stemmed_matched, ref_not_stemmed_match_h2)


        h1_meteor_score = meteor_calculate(h1_clean, h1_synonym_matched, ref_clean)
        h2_meteor_score = meteor_calculate(h2_clean, h2_synonym_matched, ref_clean)

        # LEPOR
        # h1_length_penalty = length_penalty(h1_clean, ref_clean)
        # h2_length_penalty = length_penalty(h2_clean, ref_clean)

        h1_length_penalty = 1
        h2_length_penalty = 1

        h1_score = h1_meteor_score * h1_length_penalty
        h2_score = h2_meteor_score * h2_length_penalty

        print(1 if h1_score > h2_score else # \begin{cases}
                (0 if h1_score == h2_score
                    else -1)) # \end{cases}

# convention to allow import of this file as a module
if __name__ == '__main__':
    main()
