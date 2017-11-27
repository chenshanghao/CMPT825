#!/usr/bin/env python
import optparse
import sys
import models
import copy
import time
from collections import namedtuple

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="data/input", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=50, type="int", help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-s", "--stacks-size", dest="s", default=10000, type="int", help="Maximum stack size (default=10000)")
optparser.add_option("-d", "--disorder", dest="disord", default=10, type="int", help="Disorder limit (default=10)")
# still has problem on beam search. So we didn't add detaild code. The test code is in sha_decode.
optparser.add_option("-w", "--beam width", dest="bwidth", default=5.0,  help="beamwidth")
# we didn't test this. It may be help us improve
optparser.add_option("-p", "--distortion-penalty", dest="n", default=0.0, type="float",  help="distortion penalty parameter variable")  

opts = optparser.parse_args()[0]

#Initialize language model
tm = models.TM(opts.tm, opts.k)
lm = models.LM(opts.lm)
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]
hypothesis = namedtuple("hypothesis", "lm_state, logprob, coverage, end, predecessor, phrase, french")
EngFre = namedtuple("EngFre", "english, french")


##################################
# Useful functions from score-decoder.py
def bitmap(sequence):
    """ Generate a coverage bitmap for a sequence of indexes """
    return reduce(lambda x,y: x|y, map(lambda i: long('1'+'0'*i,2), sequence), 0)

def onbits(b):
    """ Count number of on bits in a bitmap """
    return 0 if b==0 else (1 if b&1==1 else 0) + onbits(b>>1)

def prefix1bits(b):
    """ Count number of bits encountered before first 0 """
    return 0 if b&1==0 else 1+prefix1bits(b>>1)

def last1bit(b):
    """ Return index of highest order bit that is on """
    return 0 if b==0 else 1+last1bit(b>>1)

# Utility functions
def bitmap2str(b, n):
  """ Generate a length-n string representation of bitmap b """
  return bin(b)[:1:-1].ljust(n,'0')

def pos0bits(b, n):
  """Return a list of index of bits that is 0 of length-n bitmap b"""
  return [i for i, ch in enumerate(bitmap2str(b, n)) if ch == '0']

def get_list(h, output_list):
    if h.predecessor is not None:
        get_list(h.predecessor, output_list)
        output_list.append(EngFre(h.phrase.english, h.french))

# Calculate the list's probility 
def get_prob(calculate_list):
    stance = []
    for EngFre in calculate_list:
        # print EngFre
        stance += (EngFre.english.split())
    stance = tuple(stance)
    lm_state = (stance[0],)
    score = 0.0
    for word in stance[1:]:
        (lm_state, word_score) = lm.score(lm_state, word)
        score += word_score
    return score

# compare the probity
def prob_compare(now_list, current_best_list):
    current_best_list_prob = get_prob(current_best_list)
    now_list_prob = get_prob(now_list) 
    if now_list_prob > current_best_list_prob:
        return True
    else:
        return False 

# Citation.  This function is learned from former 825 student Hexiang Hu.
def get_best_adjacent_phrase(eng_list):
    while True:
        current_best_list = copy.deepcopy(eng_list)

        # insert calculating compare
        for i in range(1, len(eng_list) - 1):
            for j in range(1, i):
                now_list = copy.deepcopy(eng_list)
                now_list.pop(i)
                now_list.insert(j, eng_list[i])
                if prob_compare(now_list, current_best_list) == True:
                    current_best_list = now_list

            for k in range(i+2, len(eng_list) -1):
                now_list = copy.deepcopy(eng_list)  
                now_list.insert(k, eng_list[i])
                now_list.pop(i)
                if prob_compare(now_list, current_best_list) == True:
                    current_best_list = now_list
        
        # swap calculating compare
        for i in range(1, len(eng_list) - 2):
            for j in range(i + 1, len(eng_list)-1):
                now_list = copy.deepcopy(eng_list)
                now_list[i], now_list[j] = now_list[j], now_list[i]
                if prob_compare(now_list, current_best_list) == True:
                    current_best_list = now_list
        # replace
        for i in range(1, len(eng_list) - 1):
            if eng_list[i].french in tm:
                for engPhrase in tm[ eng_list[i].french ]:
                    now_list = copy.deepcopy(eng_list)
                    now_list[i] = EngFre(engPhrase.english, now_list[i].french)
                    if prob_compare(now_list, current_best_list) == True:
                        current_best_list = now_list

        # if there is nothing change than break the circle
        current_best_list_prob = get_prob(current_best_list) 
        eng_list_prob = get_prob(eng_list)
        if current_best_list_prob == eng_list_prob:
            return current_best_list
        else:
            print(current_best_list_prob)
            print(eng_list_prob)
            eng_list = current_best_list

def completable(b, d):
  """Return true if the distortion limit doesn't prevents a full translation"""
  return (last1bit(b) - prefix1bits(b)) <= d


def extract_english(h):
  """Return the current English string of hypothesis h"""
  return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)

def main():
    
    # calculate time for this algorithm
    start = time.time()
    # tm should translate unknown words as-is with probability 1
    for word in set(sum(french,())):
        if (word,) not in tm:
            tm[(word,)] = [models.phrase(word, 0.0)]

    total_prob = 0
    sys.stderr.write("Decoding %s...\n" % (opts.input,))

    # The following code is trying to find the best sentence
    for index,f in enumerate(french):
        # initialize hypothesis stack 
        initial_hypothesis = hypothesis(lm.begin(), 0.0, 0, 0, None, None, None)
        stacks = [{} for _ in f] + [{}]
        stacks[0][lm.begin(), 0, 0] = initial_hypothesis
        for i, stack in enumerate(stacks[:-1]):
            for h in sorted(stack.itervalues(),key=lambda h: -h.logprob)[:opts.s]: 
                fopen = prefix1bits(h.coverage)
                for j in xrange(fopen,min(fopen+1+opts.disord, len(f)+1)):
                    for k in xrange(j+1, len(f)+1):
                        if f[j:k] in tm:
                            if (h.coverage & bitmap(range(j, k))) == 0:
                                for phrase in tm[f[j:k]]:

                                    lm_prob = 0
                                    lm_state = h.lm_state

                                    for word in phrase.english.split():
                                        (lm_state, prob) = lm.score(lm_state, word)
                                        lm_prob += prob
                                    lm_prob += lm.end(lm_state) if k == len(f) else 0.0
                                    coverage = h.coverage | bitmap(range(j, k))

                                    #update the log logprob
                                    logprob = h.logprob + lm_prob + phrase.logprob 
                                    logprob += opts.n * abs(h.end + 1 - j)    #add penality

                                    #create new hypothesis
                                    new_hypothesis = hypothesis(lm_state, logprob, coverage, k, h, phrase, f[j:k])

                                    # add the new hypothesis to the stack
                                    num = onbits(coverage)
                                    if (lm_state, coverage, k) not in stacks[num] or new_hypothesis.logprob > stacks[num][lm_state, coverage, k].logprob:
                                        stacks[num][lm_state, coverage, k] = new_hypothesis

        winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)
        eng_list = [EngFre("<s>", ("<s>"))]
        get_list(winner, eng_list)
        eng_list.append(EngFre("</s>", ("</s>")))

        sys.stderr.write("Starting brute-force on setence %d ...\n" % index)

        # reordering adjacent phrases. brute-force search
        eng_list = get_best_adjacent_phrase(eng_list)

        for word in eng_list[1:-1]:
            print word.english

        sys.stderr.write("#{0}:{2} - {1}\n".format(index, ' '.join([ef.english for ef in eng_list]) , get_prob(eng_list)))
        end=time.time()
        sys.stderr.write('Elapsed Time: {}'.format(end-start))

if __name__ == "__main__":
    main()
