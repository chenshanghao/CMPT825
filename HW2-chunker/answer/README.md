# Run commands, descriptions and motivations

## Run commands

```bash
python answer/chunk.py -e 10
python perc.py -m data/default.model | python score-chunks.py
```

## Descriptions and motivations

Function perc_train in chunk.py will compare estimated tags returned from viterbi based current weights.
And then update weights for each training data point (online) like below.

```python
if tag_hat == tag_true:
    do nothing
else:
    weights[tag_true]++
    weights[tag_estimated]--
```

Function perc_test in perc.py will implement viterbi algorithms based on input weights and words.
For every word, viterbi will consider every tag in the whole tag set in input weights. And select
the most likely tag of word based on values in input weights.

We implemented bigram features and averaged weights.

* Averaged perceptron

Averaged perceptron is an approximation to the voted perceptron which is more stable compared to simple perceptron. And it  reduces space and time complexities.
Except updating weights per epoch, we aggregate weights to a dictionary sum_weights, and at last returns a avg_weights based on below equation.
`avg_weights = sum_weights/(len(train)*numepochs*numepochs).`

* Bigram

Bigram is one of the feature vector but not implemented, so we add corresponding value for each word and the accuracy is improved by about 1%, so we keep it.
if tag_hat_cur != true_tag_cur:
    weights[(tag_hat_cur, tag_hat_next)]--
    weights[(tag_true_cur, tag_true_next)]++
