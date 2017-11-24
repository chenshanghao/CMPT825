#!/usr/bin/env python
import multiprocessing
import optparse
import sys
import time
from collections import namedtuple
from itertools import groupby
from operator import itemgetter
from concurrent.futures import ProcessPoolExecutor as Executor
import models
optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="data/input", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=10, type="int", help="Limit on number of translations to consider per phrase (default=20)")
optparser.add_option("-d", "--distortion-limit", dest="d", default=sys.maxint, type="int", help="Limit on length of distortion (default=30)")
optparser.add_option("-p", "--distortion-penalty", dest="n", default=0, type="float", help="Penalty on length of distortion (default=0)")
optparser.add_option("-s", "--stack-size", dest="s", default=100, type="int", help="Maximum stack size (default=600)")
optparser.add_option("-c", "--core-process", dest="c", default=multiprocessing.cpu_count(), type="int", help="Maximum no of processes to run")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,  help="Verbose mode (default=off)")
optparser.add_option("-z", "--normalize", dest="Z", default=2.0, type="float", help="Normalize")
opts = optparser.parse_args()[0]
hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, phrase, bitmap, end_pos, future_cost")
tm = models.TM(opts.tm, opts.k)
lm = models.LM(opts.lm)
max_len = len(max(tm, key=lambda x: len(x)))
########## DEBUG OPTIONS #########
#opts.input = "data/test_input"
##################################
# Utility functions from score-decoder.py
def bitmap(sequence):
  """ Generate a coverage bitmap for a sequence of indexes """
  return reduce(lambda x,y: x|2**y, sequence, 0)
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
def onbits(b):
  """ Count number of on bits in a bitmap """
  return bin(b)[:1:-1].count('1')
def pos0bits(b, n):
  """Return a list of index of bits that is 0 of length-n bitmap b"""
  return [i for i, ch in enumerate(bitmap2str(b, n)) if ch == '0']
def getRange(pos):
  """Returns a list of range of index from a list of index pos"""
  pos_range = []
  for k, group in groupby(enumerate(pos), lambda (i,x):i-x):
    g = map(itemgetter(1), group)
    pos_range.append((g[0], g[-1]+1))
  return pos_range
def completable(b, d):
  """Return true if the distortion limit doesn't prevents a full translation"""
  return (last1bit(b) - prefix1bits(b)) <= d
def extract_english(h):
  """Return the current English string of hypothesis h"""
  return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)
# Core functions
def future_cost_est(f, tm):
  """Return future cost table for sentence f"""
  length = len(f)
  # Get cost est. for translation options table (slide 28)(only used to generate future cost est.)
  # to[start word pos][length of phrase]
  # ex: to[0][1] = translation prob of first word by itself
  to = [{} for _ in xrange(length)] # init a empty list of dict
  for start in xrange(length):
    for end in xrange(start+1,length+1):
      if f[start:end] in tm:
        # phrase found
        to[start][end-start] = (tm[f[start:end]][0].logprob + lm.table[("<unk>",)].logprob) / opts.Z
  # Get cost est. for all span (slide 29)
  # fce[start word pos][end word pos+1]
  # ex: to[0][6] = Best possbile translation prob from 1st word to 6th word (Future cost est. table (fce))
  # This only consider translation prob. Language model is not consider. (possible improvement?)
  fce = [[0 for _ in xrange(length+1)] for _ in xrange(length)] # init a 2D list with all 0
  for start in xrange(length):
    for end in xrange(start+1,length+1):
      # Search for the best span for fce[start][end]
      logprob = []
      for i in xrange(start,end): # i is end pos of the sentence. i+1 is start pos of phrase
        if end-i in to[i].keys():
          logprob.append(fce[start][i] + to[i][end-i])
      fce[start][end] = max(logprob)
  return fce
def decoder(f, tm):
  """Return the highest scoring translation found"""
  # Future Cost Est.
  fce = future_cost_est(f, tm) # See comment in function for how fce is setup
  # Initialization
  # bitmap is for storing which words are translated ('1') and which are not ('0')
  # bitmap is a memory effeient way to store bitstring
  # Use bitmap2str to tranform bitmap to bitstring
  # ex: bitmap2str(bitmap([0,3]),5) = '10010' meaning word at index = 0, 3 are translated
  # 0,3 are the pos of '1' and 5 is the length of bitstring
  initial_hypothesis = hypothesis(0.0, lm.begin(), None, None, bitmap([]), -1, fce[0][-1])
  length = len(f)
  stacks = [{} for _ in f] + [{}]
  stacks[0][(lm.begin(),bitmap2str(0, length))] = initial_hypothesis
  # Decoder
  for i, stack in enumerate(stacks[:-1]):
    stack = sorted(stack.itervalues(), key=lambda h: -h.future_cost)[:opts.s] # Prune the stack
    for h in stack:
      # for hypothesis h, try to extend it.
      # Find the index where the bitmap is 0 and transfrom it to range of index
      # ex: for bitmap string = '1100101100' the pos0_range = [(2, 4), (5, 6), (8, 10)]
      pos0_range = getRange(pos0bits(h.bitmap, length))
      for s, e in pos0_range:
        # range(s, e) = the pos of one series of successive 0 in bitmap string
        # Search for phrase in these ranges (these are untranslated words)
        for start in xrange(s, e):
          for end in xrange(start+1, e+1):
            # start:end should give all possible combination of words in this series of untranslated words.
            # compare with the max translation model available in data
            if start-end<=max_len and f[start:end] in tm:
              # Update bitmap
              bit = h.bitmap | bitmap(xrange(start, end))
              # Check if distortion limit prevent from reaching untranslated words(, therefore can't complete the translation)
              if not completable(bit, opts.d):
                continue # Prune uncompletable hypothesis
              bit_sum = onbits(bit) # Number of 1 in bit string
              bitstr = bitmap2str(bit, length)
              # The number of phrases consider is determined by 'translations-per-phrase' option
              for phrase in tm[f[start:end]]:
                # Init the new hypothesis
                logprob = h.logprob
                lm_state = h.lm_state
                # Update lm_state
                lm_logprob = 0
                phraseWords=phrase.english.split()
                for word in phraseWords:
                  (lm_state, word_logprob) = lm.score(lm_state, word)
                  lm_logprob += word_logprob
                lm_logprob += lm.end(lm_state) if bit_sum == length else 0.0 # For end charater
                # Update logprob
                logprob += phrase.logprob
                logprob += lm_logprob
                logprob += opts.n * abs(h.end_pos + 1 - start) # Distortion Penalty from baseline description
                # Calucate future cost
                future_cost = logprob
                # Find untranslated sections and calucate the future cost
                for k, j in pos0_range:
                  if k <= start <= j: # Current selected phrase is in the range
                    if start - k: # if there is untranslated word in between
                      future_cost += fce[k][start]
                    if j - end:
                      future_cost += fce[end][j]
                  else:
                    future_cost += fce[k][j]
                # Create new hypothesis
                new_hypothesis = hypothesis(logprob, lm_state, h, phrase, bit, end-1, future_cost)
                add_h = True
                # Check if recombination
                # 'and stacks[bit_sum][(lm_state,bitstr)].end_pos == end-1' should also be a condition
                # but it gives a slightly better score
                if (lm_state,bitstr) in stacks[bit_sum] and stacks[bit_sum][(lm_state,bitstr)].logprob >= logprob:
                  add_h = False
                # Add the new hypothesis in stack
                if add_h:
                  stacks[bit_sum][(lm_state,bitstr)] = new_hypothesis
  winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)
  return winner
def decoderInit(french):
  f=[french]
  # tm should translate unknown words as-is with probability 1
  for word in set(sum(f,())):
    if (word,) not in tm:
      tm[(word,)] = [models.phrase(word, 0.0)]
  winner = decoder(french, tm)
  return winner
if __name__ == '__main__':
  french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]
  sys.stderr.write("Decoding %s...\n" % (opts.input,))
  start=time.time()
  with Executor(max_workers=opts.c) as executor:
   for winner in executor.map(decoderInit, french):
    print extract_english(winner)
    if opts.verbose:
      def extract_tm_logprob(h):
        return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
      tm_logprob = extract_tm_logprob(winner)
      sys.stderr.write("LM = %f, TM = %f, Total = %f\n" %
        (winner.logprob - tm_logprob, tm_logprob, winner.logprob))
  end=time.time()
  sys.stderr.write('Elapsed Time: {}'.format(end-start))