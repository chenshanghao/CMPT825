# pylint: disable = I0011, E0401, C0103, C0321, W1401
"""
You have to write the perc_train function that trains the feature weights using the perceptron
algorithm for the CoNLL 2000 chunking task.
Each element of train_data is a (labeled_list, feat_list) pair.
Inside the perceptron training loop:
    - Call perc_test to get the tagging based on the current feat_vec and compare it with the
        true output from the labeled_list
    - If the output is incorrect then we have to update feat_vec (the weight vector)
    - In the notation used in the paper we have w = w_0, w_1, ..., w_n corresponding
        to \phi_0(x,y), \phi_1(x,y), ..., \phi_n(x,y)
    - Instead of indexing each feature with an integer we index each feature using a
        string we called feature_id
    - The feature_id is constructed using the elements of feat_list (which correspond to x above)
        combined with the output tag (which correspond to y above)
    - The function perc_test shows how the feature_id is constructed for each word in the
        input, including the bigram feature "B:" which is a special case
    - feat_vec[feature_id] is the weight associated with feature_id
    - This dictionary lookup lets us implement a sparse vector dot product where any feature_id
        not used in a particular example does not participate in the dot product
    - To save space and time make sure you do not store zero values in the feat_vec dictionary
        which can happen if \phi(x_i,y_i) - \phi(x_i,y_{perc_test}) results in a zero value
    - If you are going word by word to check if the predicted tag is equal to the true tag, there
        is a corner case where the bigram 'T_{i-1} T_i' is incorrect even though T_i is correct.
"""

from collections import defaultdict
import optparse
import os
import sys
import perc

VEC_LEN = 20
def perc_train(train, target_set, numepochs):
    '''based on train and Y_set train porper weights to get correct syntax tag'''
    weights = defaultdict(int)
    sum_weights = defaultdict(int)
    epoch_num, sentence_num = 0, 0
    for epoch in xrange(numepochs):
        epoch_num += 1
        mistakes = 0
        for sentence in train:
            sentence_num += 1
            words, words_feats = sentence
            true_y = [words[i].split()[2] for i in xrange(len(words))]
            y_hat = perc.perc_test(weights, words, words_feats, target_set, target_set[0])

            for i in xrange(len(y_hat)-1):
                if y_hat[i]!= true_y[i]: 
                    mistakes += 20
                    for j in xrange(20):
                        if (words_feats[i*20+j], y_hat[i]) in weights:
                            weights[(words_feats[i*20+j], y_hat[i])] -= 1
                        else:
                            weights[(words_feats[i*20+j], y_hat[i])] = -1
                        if (words_feats[i*20+j], y_hat[i]) in weights:
                            weights[(words_feats[i*20+j], true_y[i])] += 1
                        else:
                            weights[(words_feats[i*20+j], true_y[i])] = 1
                    weights[('B:'+y_hat[i], y_hat[i+1])] -= 1
                    weights[('B:'+true_y[i], true_y[i+1])] += 1
            for (feat, tag) in weights.iteritems():
                sum_weights[feat] += weights[feat]

        print "epoch {0} has {1} mistakes".format(epoch, mistakes)

        print("sentence_num:" + str(sentence_num))
        print("epoch_num" + str(epoch_num))

    avg_weight = defaultdict(int)
    for (feat, tag) in sum_weights.iteritems():
        avg_weight[feat] = float(sum_weights[feat])/(sentence_num * epoch_num)
    return avg_weight

if __name__ == '__main__':
    OPT_PARSER = optparse.OptionParser()
    OPT_PARSER.add_option(
        "-t", "--tagsetfile", dest="tagsetfile", default=os.path.join("data", "tagset.txt"),
        help="tagset that contains all the labels produced in the output, i.e. the y in \phi(x,y)")
    OPT_PARSER.add_option(
        "-i", "--trainfile", dest="trainfile", default=os.path.join("data", "train.txt.gz"),
        #"-i", "--trainfile", dest="trainfile", default=os.path.join("data", "train.dev"),
        help="input data, i.e. the x in \phi(x,y)")
    OPT_PARSER.add_option(
        "-f", "--featfile", dest="featfile", default=os.path.join("data", "train.feats.gz"),
        #"-f", "--featfile", dest="featfile", default=os.path.join("data", "train.feats.dev"),
        help="precomputed features for the input data, i.e. the values of \phi(x,_) without y")
    OPT_PARSER.add_option(
        "-e", "--numepochs", dest="numepochs", default=int(3),
        help="number of epochs; per epoch we iterate over all the training examples")
    OPT_PARSER.add_option(
        "-m", "--modelfile", dest="modelfile", default=os.path.join("data", "default.model"),
        help="weights for all features stored on disk")
    (OPTS, _) = OPT_PARSER.parse_args()

    # each element in the feat_vec dictionary is:
    # key=feature_id value=weight
    FEAT_VEC = {}
    TAGSET = []
    TRAIN_DATA = []

    TAGSET = perc.read_tagset(OPTS.tagsetfile)
    print >>sys.stderr, "reading data ..."
    TRAIN_DATA = perc.read_labeled_data(OPTS.trainfile, OPTS.featfile)
    print >>sys.stderr, "done."
    FEAT_VEC = perc_train(TRAIN_DATA, TAGSET, int(OPTS.numepochs))
    perc.perc_write_to_file(FEAT_VEC, OPTS.modelfile)
