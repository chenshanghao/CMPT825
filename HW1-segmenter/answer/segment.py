import sys, codecs, optparse, os
import collections, heapq, math

optparser = optparse.OptionParser()
optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('data', 'count_1w.txt'), help="unigram counts")
optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('data', 'count_2w.txt'), help="bigram counts")
optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input'), help="input file to segment")
(opts, _) = optparser.parse_args()


input_len = 0
with open(opts.input) as f:
    for l in f:
        input_len += len(unicode(l.strip(), 'utf-8'))
#print input_len
class Pdist(dict):
    "A probability distribution estimated from counts in datafile."

    def __init__(self, filename, sep='\t', N=None, missingfn=None):
        self.maxlen = 0
        for line in file(filename):
            (key, freq) = line.split(sep)
            try:
                utf8key = unicode(key, 'utf-8')
            except:
                raise ValueError("Unexpected error %s" % (sys.exc_info()[0]))
            self[utf8key] = self.get(utf8key, 0) + int(freq)
            self.maxlen = max(len(utf8key), self.maxlen)

        self.N = float(N or sum(self.itervalues()))+input_len
        self.missingfn = missingfn or (lambda k, N: len(k)*math.log10(float(1.)/N) )

    def __call__(self, key):
        if key in self:
            return math.log10(float(1+self[key])/float(self.N))#78.2->78.3
        elif len(key) <= 4:#1:80;2:83;4:83.88
            return self.missingfn(key, self.N)
        else:
            return None

# the default segmenter does not use any probabilities, but you could ...
Pw = Pdist(opts.counts1w)

#Entry for each line
Entry = collections.namedtuple('Entry', ['start_pos', 'log_prob', 'word', 'prev_entry'])
#min heap queue storing start_pos
hp = []

old = sys.stdout
sys.stdout = codecs.lookup('utf-8')[-1](sys.stdout)
# ignoring the dictionary provided in opts.counts
with open(opts.input) as f:
    for line in f:
        utf8line = unicode(line.strip(), 'utf-8')
        #print utf8line
        line_len = len(utf8line)
        chart = [None]*(line_len)
        for i in xrange(min(line_len, Pw.maxlen)):
            possible_word = utf8line[:i+1]
            if Pw(possible_word) != None:
                heapq.heappush(hp, Entry(start_pos=0, word=possible_word, log_prob=Pw(possible_word), prev_entry=None))
        while hp:
            entry = heapq.heappop(hp)
            endindex = entry.start_pos+len(entry.word)-1
            if chart[endindex] != None:
                if entry.log_prob > chart[endindex].log_prob:
                    chart[endindex] = entry
                else:
                    continue
            else:
                chart[endindex] = entry
            for j in xrange(endindex+1, line_len):
                possible_newword = utf8line[endindex+1:j+1]
                if len(possible_newword) > Pw.maxlen:
                    break
                if Pw(possible_newword) != None:
                    newentry = Entry(start_pos=endindex+1, word=possible_newword, log_prob=entry.log_prob+Pw(possible_newword), prev_entry=entry)
                    if newentry not in hp:
                        heapq.heappush(hp, newentry)

        finalindex = line_len-1
        finalentry = chart[finalindex]
        tmp = finalentry
        res = []
        while tmp.prev_entry != None:
            res = [tmp.word]+res
            tmp = tmp.prev_entry
        res = [tmp.word]+res
        print " ".join(res)

sys.stdout = old
