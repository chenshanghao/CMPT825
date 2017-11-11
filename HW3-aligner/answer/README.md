# Run commands, descriptions and motivations

## Run commands

```bash
python answer/align.py -p europarl -f de -n 100000 > output.a
head -100000 output.a > upload.a
```

## Descriptions and motivations

The core solution to this language alignment is to use training data to find a most likely from `f_i` to `e_j`.

* Baseline only considers alignment as ibm model 1. And for each pair of french and english sentence, considering every possible french word to english word and find the most common e for f among whole training corpus.

        argmaxt(L(t)) = argmax t \Sigma(s(log(Pr(f^(s)|e^(s),t))))

* First improvement is to train model independently, generate two set of parameters for French to Englsh and English to French. While decoding, pick their intersection only.

        opt_res = params(e|f) & params(f|e)
