#!/usr/bin/env python
import argparse # optparse is deprecated
from itertools import islice # slicing for iterators
from collections import defaultdict
import math
import operator
from fractions import Fraction


def word_matches(h, ref):
    return sum(1 for w in h if w in ref)

def smooth_method(p_n):
    """
    Smoothing method Add *epsilon* counts to precision with 0 counts.
    """
    epsilon = 0.10
    # alpha = 5

    return [(p_i[0] + epsilon)/ p_i[1]
        if p_i[0]==0 else p_i for p_i in p_n]

def count_clip(cad_d, ref_d):
    """Count the clip count for each ngram considering all references"""

    clip_count = 0
    for m in cad_d.keys():
        m_w = cad_d[m]
        m_max = 0
        if m in ref_d:
            m_max = max(m_max, ref_d[m])
        m_w = min(m_w, m_max)

        clip_count += m_w
    return clip_count

# def geometric_mean(precisions):
#     return (reduce(operator.mul, precisions)) ** (1.0 / len(precisions))

def brevity_penalty(c, r):
    # c: the length of the candidate translation
    # r: the effective reference corpus length
    if c > r:
        bp = 1
    else:
        bp = math.exp(1-(float(r)/c))
    return bp

def best_length_match(ref_l, cand_l):
    """Find the closest length of reference to that of candidate"""
    least_diff = abs(cand_l-ref_l)
    best = ref_l
    # for ref in ref_l:
    #     if abs(cand_l-ref) < least_diff:
    #         least_diff = abs(cand_l-ref)
    #         best = ref
    return best

def ngram_count(ref_sentence, cad_sentence, n):
  
    ref_sentence = ref_sentence
    cad_sentence = cad_sentence
    # ref_sentence = ref_sentence.strip().split()
    # cad_sentence = cad_sentence.strip().split()
    # print(ref_sentence)
    # print(cad_sentence)
    # exit()
    clipped_count = 0

    # reference sentence
    ref_ngram_dict = defaultdict(lambda: 0)
    tail_pointer = len(ref_sentence) - n + 1
    for i in range(tail_pointer):
        # print(range(tail_pointer))
        ngram_key = ' '.join(ref_sentence[i:i+n])    #.lower()    
        ref_ngram_dict[ngram_key] += 1

    # candidate sentence
    cad_ngram_dict = defaultdict(lambda: 0)
    tail_pointer = len(cad_sentence) - n + 1
    for i in range(tail_pointer):
        ngram_key = ' '.join(cad_sentence [i:i+n])    #.lower()
        cad_ngram_dict[ngram_key] += 1
    clipped_count += count_clip(cad_ngram_dict, ref_ngram_dict)

    c = len(cad_sentence)
    r = best_length_match(len(ref_sentence), c)
    limits = c - n + 1

    # if clipped_count == 0:
    #     pr = Fraction()
    # else:
    # pr = Fraction(clipped_count,limits, normalize=False) 

    # notice problem
    # print(limits)

    if limits <= 0:
        limits = 1
    pr = (clipped_count,limits)
    # print(pr)
    bp = brevity_penalty(c, r)
    return pr, bp





def BLEU(ref_sentence, cad_sentence, n):
    if n < 1:
        raise ValueError('N should bigger than 1')
    precisions_list = []

    for i_gram in range(1, n+1):
        pr, bp = ngram_count(ref_sentence, cad_sentence, i_gram)
        precisions_list.append(pr)
    # print(len(ref_sentence))
    # print(precisions_list)
    precisions_list = smooth_method(precisions_list)
    
    # print(precisions_list)
    

    weights = [0.92, 0.08, 0.00, 0.00]
    # print(precisions_list)
    value_s = []
    for i in range(n):
        if isinstance(precisions_list[i], tuple):            
            precision = Fraction(precisions_list[i][0],precisions_list[i][1])
            value_s.append(weights[i] * math.log(precision)) 
        else:
            precision = precisions_list[i]
            value_s.append(weights[i] * math.log(precision)) 
    # print(value_s)
    if len(value_s) == 0:
        return 0
    bleu_value =  bp * math.exp(math.fsum(value_s))

    return bleu_value


 
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

    # referece_sentence = "It is a guide to action that ensures that the military will forever heed Party commands."
    # candidate_sentence = "It is a guide to action which ensures that the military always obeys the commands of the party."
    # bleu_value = BLEU(referece_sentence, candidate_sentence, 4)
    # print(bleu_value)   
    # exit() 

    # ref = ['Support', "Workers'", 'Union', 'Will', 'Sue', 'City', 'Over', 'Layoffs']
    # cad = ['The', 'union', 'of', 'school', 'aid', 'is', 'suing', 'the', 'city', 'for', 'redundancies']
    # print(BLEU(ref, cad, 4))
    # exit()
 
    # note: the -n option does not work in the original code

    for h1, h2, ref in islice(sentences(), opts.num_sentences):
        rset = set(ref)
        h1_match = BLEU(ref, h1, 4)
        h2_match = BLEU(ref, h2, 4)

        # print(h1_match)
        # print(h2_match)
        # exit()

        print(1 if h1_match > h2_match else # \begin{cases}
                (0 if h1_match == h2_match
                    else -1)) # \end{cases}

# convention to allow import of this file as a module
if __name__ == '__main__':
    main()
