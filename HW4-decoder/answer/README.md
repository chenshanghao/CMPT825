# Run commands, notes and motivations
python

## Run commands

By setting stack-size = 10,000 and translations-per-phrase = 20, the python script will run about 4 hours and get the best result. We get the score -1271.058.

The higher the k and s you set, the better score you get.

The commands used to run program are as below:

```bash
python answer/decode.py -s 10000 -k 20 > output
python score-decoder.py < output
```

## Notes and motivations

* Tech details in decode.py
> 1. The algorithm is to find the most probability in the following formula. Compared to the default model, decode.py improves the translate by reordering adjacent phrases. 
	* argmax(translation model + language model + reordering + distortion penalty)
We didn't test distortion penalty since we didn't complete the better search algorithm.
The detailed fot this computing translation probability is on http://mt-class.org/jhu/slides/lecture-decoding.pdf (page 8)       
In the code we sum log probability of translation model, log probability of language model and reordering for possible phrase and push it into he stack, The optimal solution is the one with highest log probability. 
> 2. decode.py improves by setting large stacksize and translations-per-phrase. However, the larger you setting, the longer the code have to run.
We all try to use beam search to improve efficiency, but still meet some problems before deadline. So in our decode.py, we seems like use Brute-forced algorithm to get the highest probability.
> 3. We improve the decode.py by swapping adjacent phrase and add insert a seperator in sentence when calcuate the highest probability. 
> 4. A main problem in our algorithm is  we mayby get socre worse than baseline when we set low stack-size and translations-per-phrase. However, it cost long time for us to run.


* The beam-search algo in decode_sha185.py still have some problems. The detailed for this algorithm is on (http://www.phontron.com/slides/nlp-programming-en-13-search.pdf)
> 1. Instead of adding continuout phrase to next stack, the algo will try possible options with discontinuous phrases with user defined distortion limit, so the translated phrases will be reordered.
> 2. The model add distortion penality, |previous_phrase.end + 1 - new_phrase.start |, to the logprob calculation to improve the translation result.
> 3. The method calculate future cost to estimate the cost of translating the remaining words. We sort a stack by the sum logprob and future cost, to get the cheapest translation options. This method can help to improve the beam search result.
> 4. with the same configs, this code can get score -1320

