
Run
---

    python answer/segment.py | python score-segments.py

OR

    python answer/segment.py > output
    python score-segments.py -t output
    rm output

Data
----

In the data directory, you are provided with counts collected from
training data which contained reference segmentations of Chinese
sentenes.

The format of the `count_1w.txt` and `count_2w.txt` is a tab separated key followed by a count:

    __key__\t__count__

`__key__` can be a single word as in `count_1w` or two words separated by a space as in `count_2w.txt`

For bigrams the probability of the first word in a sentence w can be looked up in `count_2w.txt` as

    <S> w	Count

Math Equation

* bigram leading word w add 1 smoothing HeadProb(w)
  * w __not found__ in count_2w:

        HeadProb(w) = (1 / (INPUT_LINE_CNT+Sigma(c(w2))))^len(word)

  * w __found__ in count_2w:

        HeadProb(w) = (1+c('S', w)) / (INPUT_LINE_CNT+Sigma(c(w2)))

* unigram word w add 1 smoothing UniProb(w)
  * w __not found__ in count_1w:

        UniProb(w) = 1/(INPUT_LEN+Sigma(c(w1)))

  * w __found__ in count_1w:

        UniProb(w) = (1+w1count)/(INPUT_LEN+Sigma(c(w1)))

* bigram probality of w1, w2 BiProb(w1, w2)
  * "w1 w2" __not in__ count_2w

        BiProb(w1, w2) = 0

  * "w1 w2" __in__ count_2w

        BiProb(w1, w2) = PW2("w1 w2") / PW1(w1)

* Interpolation Smoothing for w1, w2 InterProb(w1, w2)
  * if w2 __not in__ count_1w:

        InterProb(w1, w2) = x*BiProb(w1, w2) + (1-x)*UniProb(w2)^len(w2)

  * if w2 __in__ count_1w:

        InterProb(w1, w2) = x*BiProb(w1, w2) + (1-x)*UniProb(w2)