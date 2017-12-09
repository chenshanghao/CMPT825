#!/usr/bin/env python
import argparse # optparse is deprecated
from itertools import islice # slicing for iterators
from collections import Counter
import copy
from nltk.stem.porter import *
from nltk.corpus import wordnet

def sentence_preprocess(word_list):
    lowercase_word_list = [word.lower() for word in word_list]
    # dropThe_word_list = filter(lambda word: word != "the", lowercase_word_list)
    dropCharacter_word_list = map(lambda word: word.strip(',.!?:;"$*%()[]<>/'), lowercase_word_list)
    return dropCharacter_word_list

def word_exact_matches(h, ref, h_matched):
    for i, h_w in enumerate(h):
        if (not h_matched[i]) and h_w in ref:
            h_matched[i] = True
    return h_matched

def encode_unicode(word):
    if isinstance(word, unicode) == True:
        return word.encode('utf8')
    else:
        return word

def word_stemmed_matches(h, h_exact_matched, ref):
    stemmer = PorterStemmer()
    stemmed_ref = [stemmer.stem(word.decode('utf-8')) for word in ref]
    encode_ref = map(lambda word: encode_unicode(word), stemmed_ref)
    ref_matched = [False] * len(ref)

    for i, h_w in enumerate(h):
        if not h_w:
            stemmed_word = stemmer.stem(h[i])
            encode_word = encode_unicode(stemmed_word)
            for j, r_w in enumerate(encode_ref):
                if encode_word == r_w and not ref_matched[j]:
                    h_exact_matched[i] = True
                    ref_matched[j] = True
                    ref[j] = h_w
    return h_exact_matched, ref

def word_synonym_matches(h, h_stemmed_matched, ref):
    for j, ref_word in enumerate(ref):
        ref_word_synonym = [encode_unicode(lemma.name()) for synset in wordnet.synsets(ref_word.decode('utf-8')) for lemma in synset.lemmas()]
        for i, h_W in enumerate(h):
            if not h_stemmed_matched[i] and encode_unicode(h_W) in ref_word_synonym:
                ref[j] = h_W
    return ref

def ngram(h, n):
    for i in xrange(1,n+1):
        new_h = ['']*(i-1) + h + ['']*(i-1)
        for j in xrange(len(new_h) - i):
            yield ' '.join(new_h[j:j+i])

def gleu(ref, h, n=4):

    h_n = Counter(ngram(h, n))
    r_n = Counter(ngram(ref, n))
    tpfp = sum(h_n.values())
    tpfn = sum(r_n.values())
    r_and_h = r_n & h_n
    tp = sum(r_and_h.values())  # True positives.
    return float(tp)/max(tpfp, tpfn)

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
        h1_exact_matched = word_exact_matches(h1_clean, ref_clean_h1, h1_matched)
        h2_exact_matched = word_exact_matches(h2_clean, ref_clean_h2, h2_matched)
        h1_stemmed_matched, ref_h1 = word_stemmed_matches(h1_clean, h1_exact_matched, ref_clean_h1)
        h2_stemmed_matched, ref_h2 = word_stemmed_matches(h2_clean, h2_exact_matched, ref_clean_h2)
        ref_h1 = word_synonym_matches(h1_clean, h1_stemmed_matched, ref_h1)
        ref_h2 = word_synonym_matches(h2_clean, h2_stemmed_matched, ref_h2)
        h1_match = gleu(ref_h1, h1_clean, 2)
        h2_match = gleu(ref_h2, h2_clean, 2)
        # h1_match = gleu(ref_clean, h1_clean, 2)
        # h2_match = gleu(ref_clean, h2_clean, 2)
        print(1 if h1_match > h2_match else # \begin{cases}
                (0 if h1_match == h2_match
                    else -1)) # \end{cases}
 
# convention to allow import of this file as a module
if __name__ == '__main__':
    main()
