import os
import sys

MIN_COUNT = 5 # minimum count to not be classified as a _RARE_ token

def replace_infreq_words(lines, word_counts):
    output_lines = [] # new lines to write to a new output file with the new _RARE_ tokens

    # iterate over lines in the file
    for line in lines:
        tokens = line.split() # split the line to get word and tag (e.g. 'Comparison' and 'O')
        if len(tokens) == 2 and word_counts[tokens[0]] < MIN_COUNT: # empty lines exist, so make sure this line has tokens in it, also check if the word is rare
            tokens[0] = "_RARE_"
        output_lines.append(' '.join(tokens) + "\n") # e.g. transform "Comparison O" to "_RARE_ O\n"

    return output_lines

def get_word_counts(lines):
    word_counts = {}

    # iterate over lines in the file
    for line in lines:
        tokens = line.split() # split the line to get word and tag (e.g. 'Comparison' and 'O')
        if not len(tokens) == 2: # we came across an empty line, so skip it
            continue

        # update count of the word on this line
        word = tokens[0]
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1

    return word_counts

# get command line arguments
if len(sys.argv) != 3:
    print("Requires 2 command-line arguments: training file and output file name")
    sys.exit(0)
train_fn = sys.argv[1]
out_fn = sys.argv[2]

# look through the training file to get the data we need
with open(train_fn, "r") as train_file:
    lines = train_file.readlines()
    word_counts = get_word_counts(lines) # get frequency of each word
    output_lines = replace_infreq_words(lines, word_counts) # replace words with less than a given count

# output transformed lines to a new file
with open(out_fn, "w+") as output_file:
    output_file.writelines(output_lines)
