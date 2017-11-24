#!/usr/bin/env python
import optparse
import sys
import models
import numpy
from collections import namedtuple
from math import log
import random

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="data/input", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint,
                     type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=1, type="int",
                     help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-s", "--stack-size", dest="s", default=100, type="int", help="Maximum stack size (default=1)")
optparser.add_option("-d", "--distort", dest="d", default=6, help="Distortion limit (def. 6)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False, help="Verbose mode (default=off)")
opts = optparser.parse_args()[0]

tm = models.TM(opts.tm, opts.k)
lm = models.LM(opts.lm)
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]
costs = None  # Matrix of future costs estimates. This should be initialized per sentence.

# returns the length of the english translation in h
# corresponds to the value r in collins' paper on decoding
def r(h):
  r_ = 0
  if h.predecessor is not None:
    r_ += r(h.predecessor)
  r_ += len(h.phrase.english.split()) if h.phrase is not None else 0
  return r_

# tm should translate unknown words as-is with probability 1
for word in set(sum(french, ())):
    if (word,) not in tm:
        tm[(word,)] = [models.phrase(word, 0.0)]


# from: http://mt-class.org/jhu/slides/lecture-decoding.pdf
# Constructs the best future cost table as described in the slides above.
def constructFutureCosts(f):
    minProb = -500
    sentenceLength = len(f)
    costs = numpy.empty((sentenceLength, sentenceLength))
    costs[:] = minProb      # Initialize the future costs table with really large costs.

    for col in range(0, sentenceLength):              # end
        for row in range(0, sentenceLength - col):       # start
            start = row       # Use these for accessing the french sentence list.
            end = row + col + 1

            bestEstimate = minProb
            if f[start:end] in tm:  # simple case: no reordering
                for phrase in tm[f[start:end]]:
                    if phrase.logprob > bestEstimate:
                        bestEstimate = phrase.logprob
            costs[row, col] = bestEstimate

            for i in range(0, col):   # See if a combination of previous costs is better
                # print "i, row, col: " + str(i) + ", " + str(row) + ", " + str(col)
                if costs[row, col - i - 1] + costs[row + col - i, i] > costs[row, col]:
                    costs[row, col] = costs[row, col - i - 1] + costs[row + col - i, i]
    # Debug info
    # print "setence: " + str(f)
    # for i in range(sentenceLength):
        # print str(f[i]) + ": " + str([str(costs[i, j]) for j in range(sentenceLength)])

    return costs


# 1. Go through last_hypothesis recursively, find what phrases/words have been translated.
# 2. Calculate best future cost estimate for the untranslated words. Return that.
def getFutureCostForHypothesis(hypothesis, f, start, end):
    # Find out what words have been translated by up to this hypothesis
    translated = numpy.zeros(len(f))
    translated[start:end] = 1
    # print str(translated)

    prevHypothesis = hypothesis
    while prevHypothesis is not None:
        translated[prevHypothesis.phraseStart : prevHypothesis.phraseEnd] = 1
        prevHypothesis = prevHypothesis.predecessor

    # calculate the future costs by summing all the untranslated contiguous chunks
    futureCost = 0
    i = 0
    while i < len(f):
        if translated[i] == 1:    # word location marked as translated. Move to next word.
            i += 1
        else:
            start = i
            end = 0
            while i + end < len(f) and translated[i + end] == 0:    # Find the end of untranslated chunk.
                end += 1
            futureCost += costs[start, end - 1]
            i = i + end + 1

    # print str(translated) + "(start, end): " + str((start, end)) + "; prev hyp: " + str((hypothesis.phrase, hypothesis.phraseStart, hypothesis.phraseEnd))
    # print "future cost: " + str(futureCost) + "\n"

    return futureCost

# Follow predecessor pointers back to find the hypothesis at a given depth
def getHypothesis( h, to_modify ):
    h_ = h
    child = None
    for j in range(to_modify):
      child = h_
      h_ = h_.predecessor
    return (h_,child)

# Given a list of phrases, constructs a new hypothesis where all those
# phrases are translated in the order they were passed to this function.
def create_hypothesis(h, at_end, phrases, f):
    # print "phrase: " + str(phrases)

    logprob = h.logprob
    lm_state = h.lm_state
    last_hypothesis = h
    for phrase, (start, end) in phrases:    # At most two phrases.
        logprob += phrase.logprob
        for word in phrase.english.split():
            (lm_state, word_logprob) = lm.score(lm_state, word)
            logprob += word_logprob
        if phrase == phrases[-1]:   # Add the "</s>" if this is the last phrase in the sentence.
            logprob += lm.end(lm_state) if at_end else 0.0

        # Calculate the future cost.
        futureCost = getFutureCostForHypothesis(h, f, start, end)

        # distortion, such as it is
        logprob += 0 if abs(r(h) - start + 1) <= int(opts.d) else -10*abs(r(h)+1-start) 

        new_hypothesis = hypothesis(logprob, lm_state, last_hypothesis, phrase, start, end, futureCost, [v if (i < start or i >= end) else 1 for (i,v) in enumerate(last_hypothesis.coverage)])
    last_hypothesis = new_hypothesis
    return (lm_state, new_hypothesis)

sys.stderr.write("Decoding %s...\n" % (opts.input,))
for f in french:
    # Construct the best future cost estimate table.
    costs = constructFutureCosts(f)

    sys.stderr.write("Working on sentence: %s\n" % (f,))

    hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, phrase, phraseStart, phraseEnd, futureCost, coverage")
    initial_hypothesis = hypothesis(0.0, lm.begin(), None, None, -1, -1, 0, [0 for i in range(len(f))])    # phraseStart, phraseEnd must not be None, since t[None:None]=Const sets whole array to Const value.
    stacks = [{} for _ in f] + [{}]  # stacks[i] holds hypotheses with i words decoded
    stacks[0][lm.begin()] = initial_hypothesis
    for i, stack in enumerate(stacks[:-1]):
        #print >>sys.stderr, "in stack", i, "len", len(stack)
        for h in sorted(stack.itervalues(), key=lambda h: -(h.logprob + h.futureCost))[:opts.s]:  # prune # take best opts.s entries in stack
          for start in range(len(f)):
          #for start in range(h.coverage.index(0),min([len(f),h.coverage.index(0)+opts.d])): # distortion limit
        for end in range(start+1,len(f) + 1):
            if sum(h.coverage[start:end]) == 0: # if this span is not covered
                if f[start:end] in tm:
                    #print >>sys.stderr, i,start,end,f[start:end],"in tm",h.coverage
              for phrase in tm[f[start:end]]:
                  (lm_state, new_hypothesis) = create_hypothesis(h, (sum(h.coverage) + (end - start)) == len(f), [(phrase, (start,end))], f)
            covered = sum(new_hypothesis.coverage)
            if (h.coverage.index(0) == start or f[h.coverage.index(0):start] in tm) and (lm_state not in stacks[covered] or stacks[covered][lm_state].logprob < new_hypothesis.logprob):  # second case is recombination
            #if (lm_state not in stacks[covered] or stacks[covered][lm_state].logprob < new_hypothesis.logprob):  # second case is recombination
                                        stacks[covered][lm_state] = new_hypothesis
    winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)

    def extract_english(h):
        return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)
    print extract_english(winner)
    print >>sys.stderr, extract_english(winner)

    if opts.verbose:
        def extract_tm_logprob(h):
            return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
        tm_logprob = extract_tm_logprob(winner)
        sys.stderr.write("LM = %f, TM = %f, Total = %f\n" %
                         (winner.logprob - tm_logprob, tm_logprob, winner.logprob))