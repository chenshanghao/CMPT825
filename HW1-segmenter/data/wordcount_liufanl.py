#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys, codecs, os

input_file = "/home/liufanl/Downloads/simplified_cn_small.txt"
output_file = "/home/liufanl/Downloads/count_1w_bigfile.txt"

word_count_dic = {}
output = open(output_file, "w")
with open(input_file, "r") as f:
    for line in f:
        utf8line = unicode(line.strip(), 'utf-8')
        words_list = utf8line.split(" ")
        for word in words_list:
            if word in word_count_dic:
                word_count_dic[word]+=1
            else:
                word_count_dic[word] = 1

keys = word_count_dic.keys()
values = word_count_dic.values()
for index in xrange(len(keys)):
    chn_word = keys[index].encode("utf-8")
    chn_word_count = values[index]
    output.write("%s %i\n" % (chn_word, chn_word_count))

output.close()

