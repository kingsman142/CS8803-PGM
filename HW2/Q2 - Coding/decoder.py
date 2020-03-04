import os, sys
import math
import pandas as pd
from scipy.io import loadmat

# read in the noisy string text
noisystring_file = loadmat('noisystring.mat')
noisystring = noisystring_file['noisystring'][0]
noisystring_len = len(noisystring)
print("Noisy string length: {}".format(noisystring_len))

# set up necessary variables
probs = [None] * noisystring_len
backpointers = [None] * noisystring_len

transitions = []
emissions = []
hidden_states = []

intermediate_state_names = [None, "start_firstname", "start_surname", "end_surname"]
firstnames = ["david", "anton", "fred", "jim", "barry"]
surnames = ["barber", "ilsung", "fox", "chain", "fitzwilliam", "quinceadams", "grafvonunterhosen"]

# populate transition matrix
transitions = pd.read_excel("Q2-transitions-and-emissions.xlsx", sheet_name = "transitions_renamed_new")
hidden_states = list(transitions[transitions.columns[0]])
normal_column_names = transitions.columns[1:]
transitions = transitions[normal_column_names]

# populate emission matrix
emissions = pd.read_excel("Q2-transitions-and-emissions.xlsx", sheet_name = "emissions_renamed_new", usecols = "B:AA")

# initialize step 0
probs[0] = { hidden_state : float('-inf') for hidden_state in hidden_states }
probs[0]["start_firstname"] = math.log( 1.0 / 26 )

# forward algorithm
for i in range(1, noisystring_len):
    noisystring_char = noisystring[i]
    print("Iteration {} / {}...".format(i, noisystring_len))
    for hidden_state in hidden_states:
        if probs[i] is None:
            probs[i] = dict()
            backpointers[i] = dict()
        probs[i][hidden_state] = float('-inf')
        for hidden_state_prev in hidden_states:
            prev_prob = probs[i-1][hidden_state_prev]
            transition_prob = math.log(transitions[hidden_state_prev][hidden_states.index(hidden_state)]) if transitions[hidden_state_prev][hidden_states.index(hidden_state)] > 0 else float('-inf')
            tmp = prev_prob + transition_prob
            if tmp > probs[i][hidden_state]:
                probs[i][hidden_state] = tmp
                backpointers[i][hidden_state] = hidden_state_prev
        emission_prob = math.log(emissions[noisystring_char][hidden_states.index(hidden_state)]) if emissions[noisystring_char][hidden_states.index(hidden_state)] > 0 else float('-inf')
        probs[i][hidden_state] += emission_prob

# backpointer
best_hidden_state = None
prob_max = float('-inf')
for hidden_state in hidden_states:
    if probs[noisystring_len-1][hidden_state] > prob_max:
        best_hidden_state = hidden_state
        prob_max = probs[noisystring_len-1][hidden_state]

# unpack
bp_index = noisystring_len-1
tags = [None] * (noisystring_len + 1)
while bp_index > 0:
    tags[bp_index] = best_hidden_state
    best_hidden_state = backpointers[bp_index][best_hidden_state]
    bp_index -= 1

# run back over the tags and print out the results the TAs ask for
start_index = None
rootname = None
last_firstname = None
num_errors = 0
firstname_surname_matches = {}
for firstname in firstnames:
    firstname_surname_matches[firstname] = {}
    for surname in surnames:
        firstname_surname_matches[firstname][surname] = 0

for index, tag in enumerate(tags):
    if tag in intermediate_state_names:
        continue

    # extract the name from the tag e.g. "david5" to "david" or "grafvonunterhosen15" to "grafvonunterhosen"
    name = None
    firstname = True
    if tag[0:-1] in (firstnames + surnames):
        name = tag[0:-1]
    elif tag[0:-2] in (firstnames + surnames):
        name = tag[0:-2]

    # determine whether it's firstname or surname
    if name in surnames and last_firstname is None:
        print("* ({}) found surname before firstname: {}".format(index, name))
        num_errors += 1

    # we found the first letter of a name
    if start_index is None:
        start_index = index
        rootname = name # search for this same name in the upcoming patterns -- we don't want to count david1, david2, jim3, david4, david5 as a valid pattern

    # check to make sure this is a valid pattern
    if not rootname == name:
        print("** ({}) rootname != name : {} != {}".format(index, rootname, name))
        num_errors += 1
        rootname = None

    if start_index is not None and tags[index + 1] in intermediate_state_names: # we found the last letter of a name
        if (index - start_index + 1) == len(rootname):
            if last_firstname is None: # found a firstname, now let's find a surname
                last_firstname = rootname
            else: # last time, we found a firstname, so since we just found a surname, we got a match
                firstname_surname_matches[last_firstname][rootname] += 1
                last_firstname = None
        else:
            print("*** ({}) name != len(name) : {} != {}".format(index, (index - start_index + 1), len(rootname)))
        start_index = None
        rootname = None

print("\nNum errors: {}\n".format(num_errors))

count_matrix = [
    ["", "barber", "ilsung", "fox", "chain", "fitzwilliam", "quinceadams", "grafvonunterhosen"],
    ["david", firstname_surname_matches["david"]["barber"], firstname_surname_matches["david"]["ilsung"], firstname_surname_matches["david"]["fox"], firstname_surname_matches["david"]["chain"], firstname_surname_matches["david"]["fitzwilliam"], firstname_surname_matches["david"]["quinceadams"], firstname_surname_matches["david"]["grafvonunterhosen"]],
    ["anton", firstname_surname_matches["anton"]["barber"], firstname_surname_matches["anton"]["ilsung"], firstname_surname_matches["anton"]["fox"], firstname_surname_matches["anton"]["chain"], firstname_surname_matches["anton"]["fitzwilliam"], firstname_surname_matches["anton"]["quinceadams"], firstname_surname_matches["anton"]["grafvonunterhosen"]],
    ["fred", firstname_surname_matches["fred"]["barber"], firstname_surname_matches["fred"]["ilsung"], firstname_surname_matches["fred"]["fox"], firstname_surname_matches["fred"]["chain"], firstname_surname_matches["fred"]["fitzwilliam"], firstname_surname_matches["fred"]["quinceadams"], firstname_surname_matches["fred"]["grafvonunterhosen"]],
    ["jim", firstname_surname_matches["jim"]["barber"], firstname_surname_matches["jim"]["ilsung"], firstname_surname_matches["jim"]["fox"], firstname_surname_matches["jim"]["chain"], firstname_surname_matches["jim"]["fitzwilliam"], firstname_surname_matches["jim"]["quinceadams"], firstname_surname_matches["jim"]["grafvonunterhosen"]],
    ["barry", firstname_surname_matches["barry"]["barber"], firstname_surname_matches["barry"]["ilsung"], firstname_surname_matches["barry"]["fox"], firstname_surname_matches["barry"]["chain"], firstname_surname_matches["barry"]["fitzwilliam"], firstname_surname_matches["barry"]["quinceadams"], firstname_surname_matches["barry"]["grafvonunterhosen"]],
]
s = [[str(e) for e in row] for row in count_matrix]
lens = [max(map(len, col)) for col in zip(*s)]
fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
table = [fmt.format(*row) for row in s]
print('\n'.join(table))
