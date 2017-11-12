# Run commands, descriptions and motivations

## Run commands

```bash
python answer/align.py -p europarl -f de -n 100000 > output.a
head -1000 output.a > upload.a
```

## Descriptions and motivations

The core solution to this language alignment is to use training data to find a most likely from `f_i` to `e_j`.

* baseline only considers alignment as ibm model 1. And for each pair of french and english sentence, considering every possible french word to english word and find the most common e for f among whole training corpus.

    argmaxt(L(t)) = argmax t \Sigma(s(log(Pr(f^(s)|e^(s),t))))

* First improvement is to train model symmetrically, generate optimal parameters for both french to englsh and from english to french.

    opt_res = params(e|f) & params(f|e)

* Modify the baseline from 'go over each word in a sentence' to 'go over the word set of this sentence'. The intuition behind this is simple, IBM model 1 only considers word pair, and pays no attention to the position. Thus, set will quicken training descently.
