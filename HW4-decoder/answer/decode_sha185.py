#!/usr/bin/env python
import optparse
import sys
from math import log
import models
from collections import namedtuple, defaultdict

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
lm = models.LM(opts.lm)
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

# tm should translate unknown words as-is with probability 1
for word in set(sum(french,())):
  if (word,) not in tm:
    tm[(word,)] = [models.phrase(word, 0.0)]


def compute_future_scores(f):
    floor = -500
    scores = defaultdict(lambda: defaultdict(lambda: float(floor)))
    for seq_length in range(1, len(f) + 1):
        for start in range(0, len(f) - seq_length + 1):
            end = start + seq_length
            phrase = f[start:end]
            if phrase in tm:
                score = max(i.logprob for i in tm[phrase])
                en_phrase = tuple(tm[phrase][0].english.strip().split())
                if en_phrase in lm.table:
                    score += lm.table[en_phrase].logprob
                else:
                    score = floor
                scores[start][end] = score

            for mid in range(start + 1, end):
                combined_score = (scores[start][mid] +
                                  scores[mid][end])
                if combined_score > scores[start][end]:
                    scores[start][end] = combined_score
    return scores

def future_score(untranslated, future_score_table):
    score = 0.0
    for span in untranslated:
        score += future_score_table[span[0]][span[1]]
    return score

def translated(h, f):
  return len(f) - sum(j-i for i, j in h.untranslated)

def untranslated_range(st , h, l):
  translated_range = [st]
  curr_h = h
  while curr_h.predecessor is not None:
    translated_range += [curr_h.s_t]
    curr_h = curr_h.predecessor
  st = sorted(translated_range, key=lambda a: a[0])
  if 0 < st[0][0]:
    sut = [(0, st[0][0])]
  else:
    sut = []
  for i in xrange(len(st) - 1):
    if st[i][1] != st[i+1][0]:
      sut += [(st[i][1], st[i+1][0])]
  if st[len(st) - 1][1] < l:
    sut += [(st[len(st) - 1][1], l)]
  return sut

def find_phrase(f):
  phrases = defaultdict(list)
  for i in range(len(f)):
    for j in range(i+1, len(f)+1):
      if f[i:j] in tm:
        phrases[i] += [j]
  return phrases

def get_best(stacks, ind=-1):
  try:
    return max(stacks[ind].itervalues(), key=lambda h: h.logprob)
  except ValueError:
    return get_best(stacks, ind-1)

sys.stderr.write("Decoding %s...\n" % (opts.input,))
for f in french:
  sys.stderr.write('.')
  # The following code implements a monotone decoding
  # algorithm (one that doesn't permute the target phrases).
  # Hence all hypotheses in stacks[i] represent translations of 
  # the first i words of the input sentence. You should generalize
  # this so that they can represent translations of *any* i words.
  hypothesis = namedtuple("hypothesis", "logprob, future_score, lm_state, phrase, s_t, untranslated, predecessor")
  phrases_list = find_phrase(f)
  distortion_factor = log(0.7)
  initial_hypothesis = hypothesis(0.0, 0, lm.begin(), None, (0, 0), [(0, len(f))], None)
  stacks = [{} for _ in f] + [{}]
  stacks[0][0] = initial_hypothesis
  stack_key = 0
  word_penality = -0.2
  future_score_table = compute_future_scores(f)
  for ind, stack in enumerate(stacks[:-1]):
    for h in sorted(stack.itervalues(),key=lambda h: -(h.logprob + h.future_score))[:opts.s]: # prune
      for start, end in h.untranslated:
        for i in [i for i in phrases_list.keys() if i >= start]:
          if h.predecessor is not None:
            if abs(i - h.predecessor.s_t[1]-1) > 8:
              break
          else:
            if abs(i-1)>4:
              break
          for j in [j for j in phrases_list[i] if j <= end]:
            for phrase in tm[f[i:j]]:
              logprob = h.logprob + phrase.logprob
              lm_state = h.lm_state
              for word in phrase.english.split():
                (lm_state, word_logprob) = lm.score(lm_state, word)
                logprob += word_logprob
              logprob += lm.end(lm_state) if (ind + j-i) == len(f) else 0.0
              if h.predecessor is not None:
                logprob += distortion_factor * abs(i - h.predecessor.s_t[1]-1) 
              logprob -= word_penality*len(phrase.english.split())
              new_untranslated = untranslated_range((i, j), h, len(f))
              new_hypothesis = hypothesis(logprob, future_score(new_untranslated, future_score_table), lm_state, phrase, (i, j), new_untranslated, h)
              if stack_key not in stacks[translated(new_hypothesis, f)]: # second case is recombination
                stacks[translated(new_hypothesis, f)][stack_key] = new_hypothesis 
                stack_key +=1

  winner = get_best(stacks, -1)
  def extract_english(h): 
    return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)
  print extract_english(winner)

  if opts.verbose:
    def extract_tm_logprob(h):
      return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
    tm_logprob = extract_tm_logprob(winner)
    sys.stderr.write("LM = %f, TM = %f, Total = %f\n" % 
      (winner.logprob - tm_logprob, tm_logprob, winner.logprob))

