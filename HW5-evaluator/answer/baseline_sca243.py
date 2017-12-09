#!/usr/bin/env python
import argparse # optparse is deprecated
from itertools import islice # slicing for iterators
 
def word_matches(h, ref):
    return sum(1 for w in h if w in ref)

def word_count(h):
    return float(len(h))
 
def main():
    parser = argparse.ArgumentParser(description='Evaluate translation hypotheses.')
    parser.add_argument('-i', '--input', default='data/hyp1-hyp2-ref',
            help='input file (default data/hyp1-hyp2-ref)')
    parser.add_argument('-n', '--num_sentences', default=None, type=int,
            help='Number of hypothesis pairs to evaluate')
    # note that if x == [1, 2, 3], then x[:None] == x[:] == x (copy); no need for sys.maxint
    opts = parser.parse_args()

    #Initialize 
    alpha, beta, gamma = 1.0, 0.08, 0.08
    #print("alpha: {0}, beta: {1}, gamma: {2}").format(alpha,beta,gamma)
    # we create a generator and avoid loading all sentences into a list
    def sentences():
        with open(opts.input) as f:
            for pair in f:
                yield [sentence.strip().split() for sentence in pair.split(' ||| ')]
 
    # note: the -n option does not work in the original code
    for h1, h2, ref in islice(sentences(), opts.num_sentences):
        rset = set(ref)
        h1_match = word_matches(h1, rset)
        h2_match = word_matches(h2, rset)


        #word_count
        rf_wordcount = word_count(rset)
        h1_wordcount = word_count(h1)
        h2_wordcount = word_count(h2)

        # precision / recall calculation
        h1_recall = h1_match / h1_wordcount
        h1_precision = h1_match / rf_wordcount
        h2_recall = h2_match / h2_wordcount
        h2_precision = h2_match / rf_wordcount

        # Calculate h1,h2 metor

        if h1_match == 0:
            h1_mentor = 0
        else:

            h1_coefficient = (1 - gamma * (h1_wordcount / h1_match)**beta )
            h1_mentor = h1_coefficient * (h1_precision * h1_recall) / ((1.0 - alpha) * h1_recall + alpha * h1_precision)

        if h2_match == 0:
            h2_mentor = 0
        else:
            h2_coefficient = (1 - gamma * (h2_wordcount / h2_match)**beta )
            h2_mentor = (h2_precision * h2_recall) / ((1.0 - alpha) * h2_recall + alpha * h2_precision)

        # exit()
        print(1 if h1_mentor > h2_mentor else # \begin{cases}
                (0 if h1_mentor == h2_mentor
                    else -1)) # \end{cases}
 
# convention to allow import of this file as a module
if __name__ == '__main__':
    main()
