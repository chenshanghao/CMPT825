# Run commands, notes and motivations

## Run commands

Below python script will run for about 80 minutes, during which consume 7.5G memory at most.

```bash
python answer/align.py -p europarl -f de -n 100000 > output.a
head -1000 output.a > upload.a
```

## Notes and motivations

The core solution to this language alignment is to use training data to find a most likely from `f_i` to `e_j`.

* Core is EM algorithm. Stage expectation collects counts based on already known and training data, then Maximization stage use viterbi alg. to find the most possible pair.
  * Expectation: $Pr(f|e,t) = \prod_{i=1}^{I} \sum_{j=1}^{J}t(f_i|e_j)$
  * Maximization: ${argmax}_t L(t) = {argmax}_t \sum_{s}log Pr(f^{(s)}|e^{(s)},t)$

* Baseline implements ibm model 1, trains the model to generate parameters $t(f_i|e_j)$ for each word pair, and uses viterbi algorithm to find the most possible word pair.
    \[t_k(f_i|e_j)=\sum_{s=1}^{N}\sum_{(f_i,e_j)\in(\bold{f}^{(s)},\bold{e}^{(s)})} \frac{count(f_i,e_j,\bold{f}^{(s)},\bold{e}^{(s)})}{count(e_j,\bold{f}^{(s)},\bold{e}^{(s)})}\]
    \[count(f_i, e_j, \bold{f}, \bold{e})=\frac{t_{k-1}(f_i|e_j)}{\sum_{a_i=1}^Jt_{k-1}(f_i|e_{a_i})}\]
    \[count(e_j, \bold{f,e})=\sum_{i=1}^Icount(f_i,e_j,\bold{f,e})\]
* First improvement is to train two models French->English and English->French, and then generate optimal parameters from the intersection of two models.

    $opt_{res} = params(e|f) \bigcap params(f|e)$

* Modify the baseline from 'go over each word in a sentence' to 'go over the word set of this sentence'. The intuition behind this is simple, IBM model 1 only considers word pair, and pays no attention to the position. Thus, set will quicken training decently.

* Diagonal alignment TODO: @SHUO

* IBM model 2
    * Given a French sentence $\bold{f}=f_1...f_n$ and English sentence $\bold{e}=e_1...e_m$, we model alignments of the form $\bold{a}=a_1...a_n$, where each $a_i$ takes a value from 1 to $m$, denoting the index of the English word to which the $i$th French word is aligned. Lexical translation parameters $t$ and this $a$ work together to provide a possibility for each word pair.
    \[p(\bold{f,a|e})=\prod_{i=1}^np(a_i=j|i,j,m,n)\times p(f_i|e_{a_i})\]
    \[p(a_i=j|i,j,m,n)=\frac{1}{Z_{i,m,n}}e^{\lambda h(i,j,m,n)}\]
    \[h(i,j,m,n)=-|\frac{i}{n}-\frac{j}{m}|\]
    * Use IBM model 1 output as model 2 input initialization.

* Smoothing TODO: @Shanghao

* Other experimented methods for improvement
    * Add $null$ words to the source sentence
        IBM model 1 hypothesizes one $null$ word per sentence, which causes $null$ word alignment rarely occur in the highest probability alignment. We address this problem by adding extra $null$ words to source sentence, but neither adding the same number of $null$ words to every sentence or setting the number of $null$ words be proportional to the length of the sentence generate better result compared to without extra $null$ word, so we abandon this method.

    * Initialize the parameters with log-likelihood-ratio
        To test the practicability of finding initial position that take us closer to the optimal parameter values for alignment accuracy, we use a heuristic model based on the log-likelihood-ratio ($\bold{LLR}) statistic, the statistic formula is defined as following:
       \[\sum_{f_i?\in\{f_i, \bar{f_i}\}}\sum_{e_j?\in\{e_j, \bar{e_j}\}} C(f_i?, e_j?)\times log\frac{p(f_i?|e_j?)}{p(f_i?)}\]
        In this formula, ${f_i} and ${e_j} are two words in an aligned target and source sentence pair, $C(f_i?, e_j?), p(f_i?|e_j?), p(f_i?)$ are the joint frequency and maximum likelihood estimates of the corresponding marginal and conditional probabilities of the different possible combination of ${f_i} and ${e_j} occurring and not occurring.
        For 100,000 sentences, there are more than 700,000 word pairs of ${f_i} and ${e_j}, it costs more than 3 hours to compute the statistic for each (${f_i}, ${e_j}), therefore we consider this method as a option and research other more efficient methods.
