# Personal contribution

1. Baseline coding on “想回杭州吃鸡腿”. However, I only get -1402. I cannot find the reason on 3 days. Then I increase the stack-size and translations-per-phrase, I get -1300. 
2. Then, I force the decoder to only swap neighboring phrases.
3. Performing hill-climbing on the best hypothesis before outputting it.
4. One the last two days, I add the refresh the algorithm with the following formula and get -1271.
	* argmax(translation model + language model + reordering + distortion penalty)
5. Personally, I'm confused with beam-search. I try several days on this but didn't get result. 

The commit identifiers from SFU GitLab: 
1. Baseline with big stack-size and translations-per-phrase. We get score -1300, commit ID: 9af4592d
2. The final version decode for our group. We get score -1271 commit ID: 88470e4b 

