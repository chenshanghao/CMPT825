
Your documentation
------------------

Make it so.

1. d72b97b10665f830e8329ffaa72ec32926642f53
	1. Implemented a quicker baseline. 
	2. Modify code style based on PEP8.
	3. Instead of comparing len(feats) times, compare len(sentence) per sentence.
	4. Read paper about multiple representation and we decided we'd better to avoid this.
2. d7774552f4be991755f75140dec0f4b774294ed0
	1. change codes to do key checking instead of using defaultdict __missingkey__, it will take much less time.