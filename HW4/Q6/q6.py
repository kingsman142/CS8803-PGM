import numpy as np
import pandas as pd

from scipy.io import loadmat

class CPT():
    def __init__(self, name, parents):
        self.name = name
        self.parents = parents
        self.table = np.random.rand(2 ** len(parents))

    def get_MLE(self, sample):
        return

data = loadmat("EMprinter.mat")["x"]
variables = ["fuse", "drum", "toner", "paper", "roller"] + ["burning", "quality", "wrinkled", "mult.pages", "paperjam"] # diagnoses + faults

N, D = data.shape
EPSILON = 0.01
NUM_VARIABLES = 10

oldTables = [CPT("fuse", []), CPT("drum", []), CPT("toner", []), CPT("paper", []), CPT("roller", [])] + [CPT("burning", [0]), [CPT("quality", [1, 2, 3]), [CPT("wrinkled", [0, 3]), [CPT("mult.pages", [3, 4]), [CPT("paperjam", [0, 4])]
while diff < EPSILON: # check for convergence
    newTables = [None] * NUM_VARIABLES # initialize tables for next iteration



    # calculate diff to check for convergence
    diff = 0.0
    for i in range(NUM_VARIABLES):
        if i < 5:
            diff += abs(newTables[i] - oldTables[i])
        else:
            diff += np.linalg.norm(newTables[i] - oldTables[i], ord = 1)
