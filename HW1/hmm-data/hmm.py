import os
import sys
import math

class HMM():
    def __init__(self, counts_fn):
        self.labels = set()
        self.words = set()

        self.wordtags = {}
        self.unigrams = {}
        self.bigrams = {}
        self.trigrams = {}

        self.counts_fn = counts_fn

    def fit(self):
        with open(self.counts_fn, "r") as counts_file:
            lines = counts_file.readlines()
            for line in lines:
                tokens = line.split()
                count = int(tokens[0]) # number of times this pattern occurs
                type = tokens[1] # i.e. WORDTAG, 1-GRAM, 2-GRAM, or 3-GRAM

                if type == 'WORDTAG':
                    tag = tokens[2] # e.g. O or I-GENE
                    word = tokens[3] # e.g. 'dimeric' or 'conductance'

                    # for a given tag/word combination, log how many times it occurred
                    if (tag, word) not in self.wordtags:
                        self.wordtags[(tag, word)] = 0
                    self.wordtags[(tag, word)] += count

                    # keep track of which tags are present in the text
                    if tag not in self.labels:
                        self.labels.add(tag)
                elif type == '1-GRAM':
                    tag = tokens[2]

                    # log how many times a given word (unigram) occurred
                    if tag not in self.unigrams:
                        self.unigrams[tag] = 0
                    self.unigrams[tag] += count
                elif type == '2-GRAM':
                    tag_1 = tokens[2]
                    tag_2 = tokens[3]

                    # log how many times a given bigram occurred
                    if (tag_1, tag_2) not in self.bigrams:
                        self.bigrams[(tag_1, tag_2)] = 0
                    self.bigrams[(tag_1, tag_2)] += count
                elif type == '3-GRAM':
                    tag_1 = tokens[2]
                    tag_2 = tokens[3]
                    tag_3 = tokens[4]

                    # log how many times a given trigram occurred
                    if (tag_1, tag_2, tag_3) not in self.trigrams:
                        self.trigrams[(tag_1, tag_2, tag_3)] = 0
                    self.trigrams[(tag_1, tag_2, tag_3)] += count

            # calculate naive emissions parameters
            self.emission_params = {}
            for wordtag, count in self.wordtags.items():
                tag, word = wordtag
                self.emission_params[wordtag] = math.log(self.wordtags[wordtag], 2) - math.log(self.unigrams[tag], 2)

            # calculate q values
            self.q_values = {}
            for tags, count in self.trigrams.items():
                tag1, tag2, tag3 = tags
                self.q_values[tags] = math.log(self.trigrams[tags], 2) - math.log(self.bigrams[(tag1, tag2)], 2) # TODO: (tag2, tag3) produces .3588 F1, but (tag1, tag2) produces .3867 F1

            self.labels.add("STOP")

    # only to be called by predict_baseline(...)
    def make_naive_prediction(self, word):
        # initialize variables
        label = None # the predicted tag for this word
        max_prob = None # the probability of the currently predicted label

        for tag in self.labels:
            curr_prob = self.wordtags.get((tag, word), 0) / self.unigrams.get((tag), 1e-10)
            if max_prob is None or curr_prob > max_prob:
                max_prob = curr_prob
                label = tag

        return label

    def predict_baseline(self, test_fn, out_fn):
        with open(test_fn, "r") as test_file:
            lines = test_file.readlines()
            lines = [line.strip() for line in lines] # strip all whitespace, including "\n"
            output_lines = [] # our predictions

            for test_word in lines: # each line only contains 1 token (e.g. "BACKGROUND", "primary", "care", etc.)
                # preprocessing of line to get test input
                if not len(test_word): # empty line/word
                    output_lines.append("\n") # prediction of an empty line is an empty line
                    continue

                # make prediction
                pred_tag = self.make_naive_prediction(test_word)
                if pred_tag is None: # the word was not found in any tag unigram dictionary, so it must be a _RARE_ token
                    pred_tag = self.make_naive_prediction("_RARE_")

                # log prediction
                output_lines.append("{} {}\n".format(test_word, pred_tag))

        # write all predictions to the output file
        with open(out_fn, "w+") as output_file:
            output_file.writelines(output_lines)

    # only to be called by predict_trigrams(...)
    def make_sentence_prediction(self, sentence):
        prev_scores = {(0, "*", "*"): 0.0} # default score is the empty sequence at the beginning of a sentence -- keep track of max score at each index of a sentence for a given tag sequence
        backpointers = {} # keep track of the most likely tag for each index of a sentence -- basically inverted table of prev_scores
        pred_tags = [] # the predicted tags we will return to the user

        get_possible_tags = lambda iter : self.labels if iter > 0 else "*" # get set of possible tags for a current index in the sentence

        # go through and calculate scores using the Viterbi algorithm for each tag at each index of the sentence
        num_words = len(sentence)
        for word_index in range(1, num_words+1): # iterate over the words
            # iterate over u and v, the two tag combinations, since we're using a trigram
            for u in get_possible_tags(word_index-1):
                for v in get_possible_tags(word_index):
                    max_score = None
                    max_tag = None
                    for w in get_possible_tags(word_index-2):
                        # get log probability score, which is the sum of the log probabilities of each event -- not necessarily between 0 and 1
                        prev_score = prev_scores.get((word_index-1, w, u), -1e6)
                        q = self.q_values.get((w, u, v), -1e6)
                        e = self.emission_params.get((v, sentence[word_index-1]), -1e6)
                        score = prev_score + q + e

                        # found a new most likely tag, so keep track of it
                        if max_score is None or score > max_score:
                            max_score = score
                            max_tag = w
                    # keep track of the highest score and its associated tag at this index of the sentence
                    prev_scores[(word_index, u, v)] = max_score
                    backpointers[(word_index, u, v)] = max_tag

        # collect the argmax trigram tags for the final word in the sentence so we can begin backtracking
        max_score = None
        u_max = None
        v_max = None
        for u in get_possible_tags(num_words-1):
            for v in get_possible_tags(num_words):
                # get log probability score, which is the sum of the log probabilities of each event -- not necessarily between 0 and 1
                prev_score = prev_scores.get((num_words, u, v), -1e6)
                q = self.q_values.get((u, v, "."), -1e6)
                score = prev_score + q

                # found a new most likely tag, so keep track of it
                if max_score is None or score > max_score:
                    max_score = score
                    u_max = u
                    v_max = v
        pred_tags.append(v_max)
        pred_tags.append(u_max)

        # begin backtracking through the sentence, collecting the most likely tag from backpointers and apprending it to a new list of tags
        for counter, sentence_index in enumerate(range(num_words-2, 0, -1)): # start from the end of the sentence and work backwards
            pred_tags.append(backpointers[(sentence_index+2, pred_tags[counter+1], pred_tags[counter])])
        pred_tags.reverse()
        return pred_tags

    def predict_trigrams(self, test_fn, out_fn):
        with open(test_fn, "r") as test_file:
            lines = test_file.readlines()
            lines = [line.strip() for line in lines] # strip all whitespace, including "\n"
            output_lines = [] # our predictions

            curr_sentence = []
            for word in lines: # each line only contains 1 token (e.g. "BACKGROUND", "primary", "care", etc.)
                if len(word): # not an empty line/word, so we can't start making predictions for the sentence yet
                    curr_sentence.append(word)
                else: # we have come to the end of a sentence (line is empty), so classify it
                    # make predictions given the entire sentence
                    pred_tags = self.make_sentence_prediction(curr_sentence)
                    for index, pred_tag in enumerate(pred_tags): # there is one predicted tag per word
                        output_lines.append("{} {}\n".format(curr_sentence[index], pred_tag)) # log prediction
                    output_lines.append("\n") # end of the sentence, so make a blank prediction for a blank space

                    curr_sentence = [] # reset the sentence so we can make predictions for the next sentence

        # write all predictions to the output file
        with open(out_fn, "w+") as output_file:
            output_file.writelines(output_lines)

# get command-line arguments
if len(sys.argv) != 5:
    print("Requires 4 arguments: mode ('baseline' or 'trigram'), rare counts filename, test filename, and output filename")
    sys.exit(0)
mode = sys.argv[1]
counts_fn = sys.argv[2]
test_fn = sys.argv[3]
out_fn = sys.argv[4]

# create emissions classifier and fit it on unigrams, bigrams, and trigrams
clf = HMM(counts_fn)
print("Fitting HMM parameters...")
clf.fit()

if mode == 'baseline':
    # predict for baseline HMM
    print("Making predictions for baseline HMM...")
    clf.predict_baseline(test_fn, out_fn)
else: # mode == 'trigrams' or something else
    # predict for trigram HMM
    print("Making predictions for trigram HMM...")
    clf.predict_trigrams(test_fn, out_fn)
