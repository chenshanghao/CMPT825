# Run commands, notes and motivations
python

## Run commands

By setting stack-size = 1,000 and translations-per-phrase = 20, we get total corpus log probability (LM+TM) = -1300.526262.

The commands used to run program are as below:

python answer/decodepy -s 1000 -k 20 > output
python score-decoder.py < output


## Notes and motivations

The baseline is beam-search and adjacent phrases swapping.
