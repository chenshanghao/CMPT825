yingxiuc20171008
1. Based on unigram modified program to corporate bigram
2. Add HeadAdd1Smooth to calculate prob. of leading word of each sentence
3. Add UnigramAdd1Smooth to calculate prob of unigram
4. Add BigramProb to calculate prob. of bigram.
5. Add Jelinek-Mercer smoothing function to calculate P(wi|wi-1)
6. Format code based on pylint.
merge tag: 13a8aa76d62ec9661cc7507bcd378366f287394f
push tag: 692778ebf8bb136722e45258da4249b3fe932c88


yingxiuc20170929
Initial version of hw-1.
1. Iterative DP on unigram
2. Add 1 smoothing
3. Accept at most 4-char-long unknown, but takes 1.5h to finish, needs to be refined.
push tag: 1c9ba10fd2f932fb5368448bd31bd4546390ff0d