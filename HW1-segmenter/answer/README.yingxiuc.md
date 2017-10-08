yingxiuc20171008
1. Based on unigram modified program to corporate bigram
2. Add HeadAdd1Smooth to calculate prob. of leading word of each sentence
3. Add UnigramAdd1Smooth to calculate prob of unigram
4. Add BigramProb to calculate prob. of bigram.
5. Add Jelinek-Mercer smoothing function to calculate P(wi|wi-1)
6. Format code based on pylint.
yingxiuc20170929
Initial version of hw-1.
1. Iterative DP on unigram
2. Add 1 smoothing
3. Accept at most 4-char-long unknown, but takes 1.5h to finish, needs to be refined.
