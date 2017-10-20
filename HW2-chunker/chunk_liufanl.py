"""

You have to write the perc_train function that trains the feature weights using the perceptron algorithm for the CoNLL 2000 chunking task.

Each element of train_data is a (labeled_list, feat_list) pair. 

Inside the perceptron training loop:

    - Call perc_test to get the tagging based on the current feat_vec and compare it with the true output from the labeled_list

    - If the output is incorrect then we have to update feat_vec (the weight vector)

    - In the notation used in the paper we have w = w_0, w_1, ..., w_n corresponding to \phi_0(x,y), \phi_1(x,y), ..., \phi_n(x,y)

    - Instead of indexing each feature with an integer we index each feature using a string we called feature_id

    - The feature_id is constructed using the elements of feat_list (which correspond to x above) combined with the output tag (which correspond to y above)

    - The function perc_test shows how the feature_id is constructed for each word in the input, including the bigram feature "B:" which is a special case

    - feat_vec[feature_id] is the weight associated with feature_id

    - This dictionary lookup lets us implement a sparse vector dot product where any feature_id not used in a particular example does not participate in the dot product

    - To save space and time make sure you do not store zero values in the feat_vec dictionary which can happen if \phi(x_i,y_i) - \phi(x_i,y_{perc_test}) results in a zero value

    - If you are going word by word to check if the predicted tag is equal to the true tag, there is a corner case where the bigram 'T_{i-1} T_i' is incorrect even though T_i is correct.

"""

import perc
import sys, optparse, os
from collections import defaultdict

def perc_train(train_data, tagset, numepochs):
    feat_vec = defaultdict(int)
    # insert your code here
    # please limit the number of iterations of training to n iterations
    numepochs = 1
    default_tag = "B-NP"
    FEATS_LENGTH = 20

    # train_data[i] is for i th sentence, train_data[i][0] are words with tags in train.txt, train_data[i][1] are features for all words in train.feats.dev

    for epoch in xrange(numepochs):
        mistake_num = 0
        for sentence_index in xrange(len(train_data)):
        # for sentence_index in xrange(1):
            labeled_list = train_data[sentence_index][0]
            feat_list = train_data[sentence_index][1]

            true_tag = []
            for word_tag in labeled_list:
                fields = word_tag.split()
                true_tag.append(fields[2])

            output = perc.perc_test(feat_vec, labeled_list, feat_list, tagset, default_tag)

            feat_index = 0
            # while feat_index <= len(feat_list) and tag_index < len(true_tag):
            #     (feat_index, feats) = perc.feats_for_word(feat_index, feat_list)
            #     for feature in feats:
            #         feature_id = (feature, output[tag_index])
            #         if output[tag_index] != true_tag[tag_index]:
            #             mistake_num += 1
            #             if feat_vec[feature_id] != -1:
            #                 feat_vec[feature_id] -= 1
            #         elif output[tag_index] == true_tag[tag_index] and feat_vec[feature_id] != 1:
            #             feat_vec[feature_id] += 1
            #     tag_index += 1

            for feat_index in xrange(len(feat_list)):
                tag_index = feat_index / FEATS_LENGTH # (0~19) / 20 = 0, (20~39) / 20 = 1, ....
                if output[tag_index] != true_tag[tag_index]:
                    feature_id_output = (feat_list[feat_index], output[tag_index])
                    feature_id_true = (feat_list[feat_index], true_tag[tag_index])

                    mistake_num += 1
                    feat_vec[feature_id_output] -= 1.0
                    feat_vec[feature_id_true] += 1.0

        for key, value in feat_vec.items():
            if value == 0:
                del feat_vec[key]

        print  "epoch = " + str(epoch + 1) + ", number of mistakes: " + str(mistake_num)
        print "****************************************"
    # print feat_vec

    return feat_vec

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-t", "--tagsetfile", dest="tagsetfile", default=os.path.join("data", "tagset.txt"), help="tagset that contains all the labels produced in the output, i.e. the y in \phi(x,y)")
    optparser.add_option("-i", "--trainfile", dest="trainfile", default=os.path.join("data", "train.txt.gz"), help="input data, i.e. the x in \phi(x,y)")
    optparser.add_option("-f", "--featfile", dest="featfile", default=os.path.join("data", "train.feats.gz"), help="precomputed features for the input data, i.e. the values of \phi(x,_) without y")
    optparser.add_option("-e", "--numepochs", dest="numepochs", default=int(10), help="number of epochs of training; in each epoch we iterate over over all the training examples")
    optparser.add_option("-m", "--modelfile", dest="modelfile", default=os.path.join("data", "default.model"), help="weights for all features stored on disk")
    (opts, _) = optparser.parse_args()

    # each element in the feat_vec dictionary is:
    # key=feature_id value=weight
    feat_vec = {}
    tagset = []
    train_data = []

    tagset = perc.read_tagset(opts.tagsetfile)
    print >>sys.stderr, "reading data ..."
    train_data = perc.read_labeled_data(opts.trainfile, opts.featfile)
    print >>sys.stderr, "done."
    feat_vec = perc_train(train_data, tagset, int(opts.numepochs))
    perc.perc_write_to_file(feat_vec, opts.modelfile)

