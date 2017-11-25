# Run commands, notes and motivations

## Run commands

By setting stack-size = 1,000 and translations-per-phrase = 20, we get total corpus log probability (LM+TM) = -1300.526262.

The commands used to run program are as below:

python answer/decodepy -s 1000 -k 20 > output
python score-decoder.py < output


## Notes and motivations


* Tech details in decode.py
1. The algorithm is to find arg\max_{e}(\max_{a}logPrTM(f,a∣e)+logPrLM(e)). In the code we sum log probability of translation model and log probability of language model for each possible phrase and push it the jth stack, where j is the lenth of translated words. The optimal solution is the one with highest log probability.
2. decode.py improves the translate by reordering adjacent phrases. 

* Another beam-search algo in decode_sha185.py
1. Instead of adding continuout phrase to next stack, the algo will try possible options with discontinuous phrases with user defined distortion limit, so the translated phrases will be reordered.
2. The model add distortion penality, |previous_phrase.end + 1 - new_phrase.start |, to the logprob calculation to improve the translation result.
3. The method calculate future cost to estimate the cost of translating the remaining words. We sort a stack by the sum logprob and future cost, to get the cheapest translation options. This method can help to improve the beam search result.
4. with the same configs, this code can get score -1320
