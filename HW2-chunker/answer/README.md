
Your documentation
------------------

Make it so.

python answer/chunk.py -e 10
python perc.py -m data/default.model | python score-chunks.py

##Implementation Details

Function perc_train in chunk.py will compare estimated tags returned from viterbi based current weights.
And then update weights for each training data point (online) like below.
if tag_hat == tag_true:
	do nothing
else:
	weights[tag_true]++
	weights[tag_estimated]--

Function perc_test in perc.py will implement viterbi algorithms based on input weights and words.
For every word, viterbi will consider every tag in the whole tag set in input weights. And select
the most likely tag of word based on values in input weights.

We implemented bigram features and averaged weights.
1. For bigram
if tag_hat_cur != true_tag_cur:
    weights[(tag_hat_cur, tag_hat_next)]--
    weights[(tag_true_cur, tag_true_next)]++
2. Average weights
Except updating weights by epoch, we aggregate weights per epoch in a sum_weights, and return sum_weights/(len(train)*numepochs*numepochs).