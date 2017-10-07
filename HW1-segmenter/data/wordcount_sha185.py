#! /usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter

word = Counter()
biword = Counter()
with open('wseg_simplified_cn.txt', 'rb') as lines:
    for line in lines:
        words = line.strip().split()
        word.update({i: 1 for i in words})
        biword.update({i: 1 for i in zip(words[:-1], words[1:])})

with open('count1.txt', 'wb') as output1:
    for k, v in word.iteritems():
        output1.write('{}\t{}\n'.format(k, v))
with open('count2.txt', 'wb') as output2:
    for (k1, k2), v in biword.iteritems():
        output2.write('{} {}\t{}\n'.format(k1, k2, v))
