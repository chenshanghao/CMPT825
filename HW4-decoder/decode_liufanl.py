#!/usr/bin/env python
import optparse
import sys
import models
from collections import namedtuple, defaultdict
from itertools import groupby
from operator import itemgetter
import heapq

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="data/input", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=1, type="int", help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-s", "--stack-size", dest="s", default=1, type="int", help="Maximum stack size (default=1)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,  help="Verbose mode (default=off)")
opts = optparser.parse_args()[0]

tm = models.TM(opts.tm, opts.k)
# tm =  {('se', 'est', '-', 'il', 'pass\xc3\xa9'): [phrase(english='has happened', logprob=0.0)], ('pos\xc3\xa9e',): [phrase(english='asked', logprob=-0.261521458626)], ('le', 'cours', 'de', 'les', 'deux', 'prochaines'): [phrase(english='the next two', logprob=0.0)],
lm = models.LM(opts.lm)
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]
# [('honorables', 's\xc3\xa9nateurs', ',', 'que', 'se', 'est', '-', 'il', 'pass\xc3\xa9', 'ici', ',', 'mardi', 'dernier', '?'), ('un', 'Comit\xc3\xa9', 'de', 's\xc3\xa9lection', 'a', '\xc3\xa9t\xc3\xa9', 'constitu\xc3\xa9', '.')]

weights = (1.0, 1.0, 1.0, 1.0)
# tm should translate unknown words as-is with probability 1
for word in set(sum(french,())):
  if (word,) not in tm:
      tm[(word,)] = [models.phrase(word, (0.0, 0.0, 0.0, 0.0))]

best_sentence = defaultdict(list)
sys.stderr.write("Decoding %s...\n" % (opts.input,))
count = 1

for f in french:
  # The following code implements a monotone decoding
  # algorithm (one that doesn't permute the target phrases).
  # Hence all hypotheses in stacks[i] represent translations of 
  # the first i words of the input sentence. You should generalize
  # this so that they can represent translations of *any* i words.
  count = count + 1
  future_cost_table = dict()

  for phrase_length in range(1, len(f)+1):
    start_positions = range(0, len(f)-phrase_length+1)
    for start in start_positions:
      end = start + phrase_length
      future_cost_table[(start, end)] = 1
      current_phrase = tm.get(f[start, end], [])
      if current_phrase:
        logprob = 0
        state = tuple()
        current_phrase_words = current_phrase[0].english.split()
        for word in current_phrase_words:
          (lm_state, word_logprob) = lm.score(lm_state, word)
          logprob += word_logprob

        all_features = list(current_phrase[0].features) + [logprob]
        score = 0
        for i in xrange(len(weights)):
          weight = weights[i]
          feature = all_features[i]
          score += weight * feature
        future_cost_table[(start, end)] = score

      for middle in range(start+1, end):
        future_cost_table[(start, end)] = max(future_cost_table[(start, middle)] + future_cost_table[(middle, end)], future_cost_table[(start, end)])

  hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, phrase")
  initial_hypothesis = hypothesis((0.0, 0.0, 0.0, 0.0), 0.0, future_cost_table[(0, len(f))], lm.begin(), None, None)
  stacks = [{} for _ in f] + [{}]
  stacks[0][lm.begin()] = initial_hypothesis
  for i, stack in enumerate(stacks[:-1]):
    for h in sorted(stack.itervalues(),key=lambda h: -h.logprob)[:opts.s]: # prune
        # h = hypothesis(logprob=0.0, lm_state=('<s>',), predecessor=None, phrase=None)
      for j in xrange(i+1,len(f)+1):
        if f[i:j] in tm:
          for phrase in tm[f[i:j]]:
            logprob = h.logprob + phrase.logprob
            lm_state = h.lm_state
            for word in phrase.english.split():
              (lm_state, word_logprob) = lm.score(lm_state, word)
              logprob += word_logprob
            logprob += lm.end(lm_state) if j == len(f) else 0.0
            new_hypothesis = hypothesis(logprob, lm_state, h, phrase)
            if lm_state not in stacks[j] or stacks[j][lm_state].logprob < logprob: # second case is recombination
              stacks[j][lm_state] = new_hypothesis 
  winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)
  def extract_english(h): 
    return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)
  print extract_english(winner)

  if opts.verbose:
    def extract_tm_logprob(h):
      return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
    tm_logprob = extract_tm_logprob(winner)
    sys.stderr.write("LM = %f, TM = %f, Total = %f\n" % 
      (winner.logprob - tm_logprob, tm_logprob, winner.logprob))

