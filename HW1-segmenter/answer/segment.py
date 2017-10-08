'''
segmenter for 2 word
'''
import codecs
import collections
import heapq
import math
import optparse
import os
import sys

OPTPARSER = optparse.OptionParser()
OPTPARSER.add_option("-c", "--unigramcounts",
                     dest='counts1w',
                     default=os.path.join('data', 'count_1w.txt'),
                     help="unigram counts")
OPTPARSER.add_option("-b", "--bigramcounts",
                     dest='counts2w',
                     default=os.path.join('data', 'count_2w.txt'),
                     help="bigram counts")
OPTPARSER.add_option("-i", "--inputfile",
                     dest="input",
                     default=os.path.join('data', 'input'),
                     help="input file to segment")
(OPTS, _) = OPTPARSER.parse_args()

INPUT_LEN = 0
with open(OPTS.input) as f:
    for l in f:
        INPUT_LEN += len(unicode(l.strip(), 'utf-8'))

INPUT_LINE_CNT = 0
with open(OPTS.input) as f:
    for l in f:
        INPUT_LINE_CNT += 1

#print INPUT_LEN
class Pdist(dict):
    "A probability distribution estimated from counts in datafile."

    def __init__(self, filename, sep='\t', N=None, missingfn=None):
        self.maxlen = 0
        for per_line in file(filename):
            (key, freq) = per_line.split(sep)
            try:
                utf8key = unicode(key, 'utf-8')
            except:
                raise ValueError("Unexpected error %s" % (sys.exc_info()[0]))
            self[utf8key] = self.get(utf8key, 0)+int(freq)
            if u' ' in utf8key:
                self.maxlen = max(len(utf8key.split()[0]), len(utf8key.split()[1]), self.maxlen)
            else:
                self.maxlen = max(len(utf8key), self.maxlen)
        self.total = float(N or sum(self.itervalues()))
        self.missingfn = missingfn or (lambda k, total: len(k))

    def __call__(self, key):
        if key in self:
            return self[key]
        else:
            return None

# the default segmenter does not use any probabilities, but you could ...
PW1 = Pdist(OPTS.counts1w)
PW2 = Pdist(OPTS.counts2w)

#Entry for each line
Entry = collections.namedtuple('Entry', ['start_pos', 'log_prob', 'word', 'prev_entry'])
#min heap queue storing start_pos
hp = []
old = sys.stdout
sys.stdout = codecs.lookup('utf-8')[-1](sys.stdout)
# ignoring the dictionary provided in OPTS.counts
def HeadAdd1Smooth(w):
    if PW2(u'<S> '+w) is None:
        return (1.0/(INPUT_LINE_CNT+PW2.total))**len(w)
    else:
        return (1.0+PW2(u'<S> '+w))/(INPUT_LINE_CNT+PW2.total)

def UnigramAdd1Smooth(w):
    '''P(w)=1+C(w)/V+C(w)
    '''
    if PW1(w) is None:
        w1count = 0
    else: w1count = PW1(w)*1.0
    return ((1+w1count)/(INPUT_LEN+PW1.total), w1count)

def BigramProb(w2):
    '''prev next
    P(next|prev)=C(prev,next)/C(prev)
    '''
    prev_w = w2.split()[0]
    if PW2(w2) is None:
        return 0
    #if w2 exist in pw2, prev and next will definitely appear in pw1
    w2count = PW2(w2)*1.0
    w1count = PW1(prev_w)*1.0
    return w2count/w1count

def JMSmooth(w2):
    '''P(w2|w1)=lambda*P(w2|w1)+(1-lambda)*P(w2)
    '''
    jm_x = 0.9
    #jm_prev = w2.split()[0]
    jm_next = w2.split()[1]
    unigram_res = UnigramAdd1Smooth(jm_next)
    if unigram_res[1] == 0:
        prob = jm_x*BigramProb(w2)+(1.0-jm_x)*(unigram_res[0]**len(jm_next))
    else:
        prob = jm_x*BigramProb(w2)+(1.0-jm_x)*(unigram_res[0])
    return math.log10(prob)

def w2len(w):
    w = w.split()
    if w[0] == u'<S>':
        return len(w[1])
    else:
        return len(w[0])+len(w[1])

with open(OPTS.input) as f:
    for line in f:
        utf8line = unicode(line.strip(), 'utf-8')
        #print utf8line
        line_len = len(utf8line)
        max_len = min(line_len, PW2.maxlen)
        chart = [None]*(line_len)

        for i in xrange(1, 4+max_len):#'<S> ' len is 4
            head = utf8line[:i]
            if len(head) <= 4:
                heapq.heappush(hp, Entry(start_pos=0,
                                         word=u'<S> '+head,
                                         log_prob=math.log10(HeadAdd1Smooth(head)),
                                         prev_entry=None))

        while hp:
            entry = heapq.heappop(hp)
            endindex = entry.start_pos+w2len(entry.word)-1
            if chart[endindex] != None:
                if entry.log_prob > chart[endindex].log_prob:
                    chart[endindex] = entry
                else:
                    continue
            else:
                chart[endindex] = entry

            entry_word2 = entry.word.split()[1]
            for j in xrange(endindex+1, line_len):
                next_word = entry_word2+u' '+utf8line[endindex+1:j+1]
                if j-endindex > PW2.maxlen:
                    break
                if j-endindex <= 5:
                    newentry = Entry(start_pos=endindex-len(entry_word2)+1,
                                     word=next_word,
                                     log_prob=entry.log_prob+JMSmooth(next_word),
                                     prev_entry=entry)
                    if newentry not in hp:
                        heapq.heappush(hp, newentry)

        finalindex = line_len-1
        finalentry = chart[finalindex]
        tmp = finalentry
        res = []
        while tmp.prev_entry != None:
            res = [tmp.word.split()[1]]+res
            tmp = tmp.prev_entry
        res = [tmp.word.split()[1]]+res
        print ' '.join(res)

sys.stdout = old
