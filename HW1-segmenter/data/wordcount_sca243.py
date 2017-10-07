#!/usr/bin/python

import os

#global parameter
DELIMITER = " " 
wordcount = {}


if __name__=='__main__':

	with open("wseg_simplified_cn.txt","r+") as f:
		for line in f:
			for word in line.split():
				word = word.lstrip()
				word = word.rstrip()
				if word not in wordcount:
					print(word)
					wordcount[word] = 1
				else:
					wordcount[word] += 1

	wordcount = sorted(wordcount.iteritems(), key=lambda d:d[1], reverse = True)

	with open("wordcount.txt","a") as f:
		for value in wordcount:
			output_str = DELIMITER.join([str(value[0]),str(value[1])])
			f.write((output_str))
			f.write("\n")