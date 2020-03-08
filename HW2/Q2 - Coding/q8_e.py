import numpy as np
from scipy.io import loadmat

class Factor():
    def __init__(self, table, condVars, inputVars, givenVars, givenVals, symptomsInit = False, diseasesInit = False, summerInit = False):
        self.table = table
        self.condVars = condVars
        self.inputVars = inputVars
        self.allVars = self.condVars + self.inputVars

        diseaseIndices = [1, 2, 3, 4]
        self.diseasesLeft = [index for index in diseaseIndices if not index in givenVars]

        self.convert_matlab_table()

        if symptomsInit:
            # only keep the dimensions that are not given
            if diseaseIndices[0] in givenVars:
                givenVarIndex = givenVars.index(diseaseIndices[0])
                self.table = self.table[givenVals[givenVarIndex]]
                self.condVars.remove(diseaseIndices[0])
            if diseaseIndices[1] in givenVars:
                givenVarIndex = givenVars.index(diseaseIndices[1])
                self.table = self.table[:, givenVals[givenVarIndex]]
                self.condVars.remove(diseaseIndices[1])
            if diseaseIndices[2] in givenVars:
                givenVarIndex = givenVars.index(diseaseIndices[2])
                self.table = self.table[:, :, givenVals[givenVarIndex]]
                self.condVars.remove(diseaseIndices[2])
            if diseaseIndices[3] in givenVars:
                givenVarIndex = givenVars.index(diseaseIndices[3])
                self.table = self.table[:, :, :, givenVals[givenVarIndex]]
                self.condVars.remove(diseaseIndices[3])
            if inputVars[0] in givenVars:
                givenVarIndex = givenVars.index(inputVars[0])
                self.table = self.table[:, :, :, :, givenVals[givenVarIndex]]
                self.inputVars.remove(inputVars[0])
        elif diseasesInit:
            # only keep the dimensions that are not given
            if 0 in givenVars:
                givenVarIndex = givenVars.index(0)
                self.table = self.table[givenVals[givenVarIndex]]
                self.condVars.remove(0)
            if inputVars[0] in givenVars:
                givenVarIndex = givenVars.index(inputVars[0])
                self.table = self.table[:, givenVals[givenVarIndex]]
                self.inputVars.remove(inputVars[0])
        elif summerInit:
            if 0 in givenVars:
                self.table = np.array([self.table[givenVals[0]]])
                self.condVars.remove(0)

        self.allVars = self.condVars + self.inputVars
        self.table = self.table.squeeze()

    def mult_factor_symptom(self, newFactor):
        messageTable = self.table
        newTable = newFactor.get_table()

        if len(self.table.shape) == len(self.diseasesLeft) and len(newTable.shape) == len(self.diseasesLeft):
            newMessageTable = messageTable
            newVars = self.diseasesLeft
        else:
            newMessageTable = np.stack((messageTable, messageTable), axis = -1) # converts from 2x2x2x2x2 to 2x2x2x2x2x2 (1 more dimension)
            newVars = list(self.diseasesLeft)
            newVars += self.allVars if not len(self.table.shape) == len(self.diseasesLeft) else []
            newVars += newFactor.allVars if not len(newTable.shape) == len(self.diseasesLeft) else []
        newVars = list(set(newVars))

        for i in range(0, int(2**len(newVars))):
            assignments = toBinaryArray(i, len(newVars)) #len(self.diseasesLeft)) # currently storing assignments for x1, x2, x3, x4, and x10
            diseaseAssignments = assignments[0:len(self.diseasesLeft)] # currently storing assignments for x1, x2, x3, x4

            currTable = newTable
            for assignment in diseaseAssignments: # traverse the table to get this combination's assignment
                currTable = currTable[assignment] # newTable[x1][x2][x3][x4]
            newFalseTrueVars = []
            if len(newTable.shape) == len(self.diseasesLeft): # this table is a table with a given variable (e.g. x8), so it doesn't have a true and false column, just a true column (its shape is 2x2x2x2, not 2x2x2x2x2)
                newFalseTrueVars.append(currTable)
                newFalseTrueVars.append(currTable)
            else:
                newFalseTrueVars.append(currTable[0]) # x10 = false
                newFalseTrueVars.append(currTable[1]) # x10 = true

            currNewMessageTable = newMessageTable
            for assignment in assignments[:-2]:
                currNewMessageTable = currNewMessageTable[assignment]
            if len(self.table.shape) == len(self.diseasesLeft):
                currNewMessageTable = currNewMessageTable[assignments[-2]]
                currNewMessageTable[assignments[-1]] *= newFalseTrueVars[0]
            else:
                # TODO: control for the case when we have a 2x2x2x2x1 and a 2x2x2x2x2 table
                # multiplying a 2x2x2x2x2 table by a 2x2x2x2x2 table
                #currNewMessageTable = currNewMessageTable[assignments[-1]]
                #print(i, currNewMessageTable.shape)
                for val in range(0, len(currNewMessageTable)): # is it 1 or 2?
                    currNewMessageTable[val][0] *= newFalseTrueVars[0]
                    currNewMessageTable[val][1] *= newFalseTrueVars[1]

        # handle some logistics
        self.condVars = newVars
        self.inputVars = []
        self.allVars = newVars
        self.table = newMessageTable.squeeze()
        assert(len(self.allVars) == len(self.table.shape))

    def mult_factor_disease(self, newFactor, marginalizedVar):
        newVars = set(self.get_vars()).union(set(newFactor.get_vars())) # combine [0, 4] and [1, 2, 3, 4] into set(0, 1, 2, 3)
        newVars.remove(marginalizedVar)
        newVars = sorted(list(newVars)) # convert from set(0, 1, 2, 3) to [0, 1, 2, 3]

        newFactorTable = np.zeros((2, ) * len(newVars))
        for i in range(0, 2**(len(newVars))): # from 0 to 7 inclusive
            assignments = toBinaryArray(i, len(newVars)) # each variable in this assignment corresponds to the respective variable in newVars
            varAssignments = {}
            for index, assignment in enumerate(assignments):
                varAssignments[newVars[index]] = assignment # place all variable assignments into a collective dict

            marginalizedVarFalseAssignment = dict(varAssignments) # create variable assignment for when the marginalized variable is false
            marginalizedVarTrueAssignment = dict(varAssignments) # create variable assignment for when the marginalized variable is true
            marginalizedVarFalseAssignment[marginalizedVar] = 0
            marginalizedVarTrueAssignment[marginalizedVar] = 1

            newFactorFalse = newFactor.getProb(marginalizedVarFalseAssignment) # disease table probability when marginalized variable is false
            newFactorTrue = newFactor.getProb(marginalizedVarTrueAssignment) # disease table prob when marginalized variable is true
            thisFalse = self.getProb(marginalizedVarFalseAssignment) # messageTable probability when marginalized variable is false
            thisTrue = self.getProb(marginalizedVarTrueAssignment) # messageTable probability when marginalized variable is true

            newProb = newFactorFalse * thisFalse + newFactorTrue * thisTrue # sum over the marginalized variable, and for each summation, there is one multiplication

            # traverse the new factor table and insert this newly calculated probability
            currTable = newFactorTable
            for assignment in assignments[:-1]:
                currTable = currTable[assignment]
            currTable[assignments[-1]] = newProb

        # handle some higher-level logistics
        self.condVars = newVars
        self.inputVars = []
        self.allVars = newVars
        self.table = newFactorTable.squeeze()
        assert(len(self.allVars) == len(self.table.shape))

    def mult_factor_multiple(self, newFactors, marginalizedVar):
        if len(newFactors) == 1: # special case, only happens when the query is P(x0)
            self.condVars = newFactors[0].get_vars()
            self.inputVars = []
            self.allVars = self.condVars
            self.table = newFactors[0].get_table()

        newVars = set(newFactors[0].get_vars())
        for factor in newFactors:
            newVars = newVars.union(factor.get_vars())
        if marginalizedVar in newVars:
            newVars.remove(marginalizedVar)
        newVars = sorted(list(newVars))

        newFactorTable = np.zeros((2, ) * len(newVars))
        for i in range(2**len(newVars)):
            assignments = toBinaryArray(i, len(newVars))
            varAssignments = {}
            for index, assignment in enumerate(assignments):
                varAssignments[newVars[index]] = assignment # place all variable assignments into a collective dict

            marginalizedVarFalseAssignment = dict(varAssignments) # create variable assignment for when the marginalized variable is false
            marginalizedVarTrueAssignment = dict(varAssignments) # create variable assignment for when the marginalized variable is true
            marginalizedVarFalseAssignment[marginalizedVar] = 0
            marginalizedVarTrueAssignment[marginalizedVar] = 1

            newProbFalse = 1.0
            newProbTrue = 1.0
            for factor in newFactors:
                factorFalse = factor.getProb(marginalizedVarFalseAssignment) # disease table probability when marginalized variable is false
                factorTrue = factor.getProb(marginalizedVarTrueAssignment) # disease table prob when marginalized variable is true

                newProbFalse *= factorFalse
                newProbTrue *= factorTrue

            newProb = newProbFalse + newProbTrue # sum over the marginalized variable, and for each summation, there is one multiplication

            # traverse the new factor table and insert this newly calculated probability
            currTable = newFactorTable
            for assignment in assignments[:-1]:
                currTable = currTable[assignment]
            currTable[assignments[-1]] = newProb

        # handle some higher-level logistics
        self.condVars = newVars
        self.inputVars = []
        self.allVars = newVars
        self.table = newFactorTable.squeeze()
        assert(len(self.allVars) == len(self.table.shape))

    def getProb(self, varAssignments):
        if self.table.size == 1:
            return self.table

        assignments = []
        for var in self.allVars: # if vars = [1, 2, 3, 4], then map varAssignments {4: 1, 0: 1, 2: 0, 3: 1, 1: 0} to [0, 0, 1, 1]
            assignments.append(varAssignments[var])

        # iterate over the table and find the probability that matches the corresponding variable assignments
        currTable = self.table
        for assignment in assignments:
            currTable = currTable[assignment]

        return currTable

    def convert_matlab_table(self):
        newTable = np.zeros((2, ) * (len(self.allVars))) # convert the table from 16x2 to 2x2x2x2x2

        if len(self.allVars) == 1: # special case: isSummer table
            newTable[0] = self.table[0]
            newTable[1] = self.table[1]
            self.table = newTable
            return

        for i in range(0, newTable.size): # loop through all the possible 16 states
            assignments = toBinaryArray(i, len(newTable.shape)) # convert from i = 5 to [0, 1, 1]

            currTable = newTable
            for assignment in assignments[:-1]: # traverse to currTable[0][1]
                currTable = currTable[assignment]
            condAssignmentToDecimal = toDecimal(assignments[:-1]) # convert [x1, x2, x3, x4] to the appropriate decimal value
            currTable[assignments[-1]] = self.table[condAssignmentToDecimal][assignments[-1]] # assign currTable[0][1][1] = self.table[5]
        self.table = newTable

    def get_table(self):
        return self.table

    def get_vars(self):
        return self.allVars

    def print_table(self):
        for i in range(0, self.table.size):
            binary = toBinaryArray(i, len(self.table.shape))
            currTable = self.table
            for assignment in binary:
                currTable = currTable[assignment]
            print(binary, currTable)

    def normalize_table(self):
        sum_probs = np.sum(self.table)
        self.table /= sum_probs

def toBinaryArray(number, length = None):
    if length is not None:
        return [int(ch) for ch in list(format(number, 'b').zfill(length)[::-1])] # all elements are ints e.g. [1, 0, 1, 0, 0, 0, 0, 0] for int(5) -- LSB is at index 0
    else:
        return [int(ch) for ch in list(format(number, 'b')[::-1])]

def toDecimal(bitArray):
    val = 0
    for index, bit in enumerate(bitArray):
        val += bit * (2**index)
    return int(val)

assignments = np.zeros((12, 1))
givens = [] # user input -- the nodes that are given
givenVals = [] # user input -- binary values, the assignments of the nodes in the givens list
queries = [] # user input -- the nodes we are querying
which_query = 1
if which_query == 1:
    givens = [8, 11]
    givenVals = [1, 1]
    queries = [1]
elif which_query == 2:
    givens = [4]
    givenVals = [1]
    queries = [7, 8, 9, 10, 11]
elif which_query == 3:
    givens = [0]
    givenVals = [1]
    queries = [10]
diseasesIndices = [1, 2, 3, 4]
for g in givens:
    assignments[g] = 1
data = loadmat("q8_vars.mat")
symptoms = data["symptoms"]
diseases = data["diseases"]
summerTable = data["summer_probs"]

variablesLeft = [1, 2, 3, 4]
messageTable = None

remainingFactors = []

# combine the symptoms tables
for i in range(5, 12): # 5 to 11 inclusive
    if not i in givens + queries: # this symptom will be marginalized out, so skip this step
        continue

    newFactor = Factor(symptoms[i-5], condVars = list(diseasesIndices), inputVars = [i], givenVars = givens, givenVals = givenVals, symptomsInit = True)

    if messageTable is None: # initialize messageTable
        messageTable = newFactor
        continue # go to next iteration/factor

    messageTable.mult_factor_symptom(newFactor)

# combine the diseases tables
for i in range(4, 0, -1):
    newFactor = Factor(diseases[i-1], condVars = [0], inputVars = [i], givenVars = givens, givenVals = givenVals, diseasesInit = True)

    if i in queries + givens:
        remainingFactors.append(newFactor)
        continue

    if messageTable is None: # in the off chance we marginalized away all the symptoms, then the message table will begin as a disease table
        messageTable = newFactor
        continue

    messageTable.mult_factor_disease(newFactor, i)

# combine the isSummer node
for i in [0]:
    newFactor = Factor(summerTable, condVars = [0], inputVars = [], givenVars = givens, givenVals = givenVals, summerInit = True)
    remainingFactors.append(newFactor)

    if messageTable is None:
        messageTable = newFactor # really weird scenario where no variables are given, so they're all marginalized out and all we get is P(x0)
        continue

remainingFactors.append(messageTable)
messageTable.mult_factor_multiple(remainingFactors, 0) # marginalize over 0

messageTable.normalize_table()
print("{} <-- variables in order".format(messageTable.get_vars()))
messageTable.print_table()
print("Sum:", np.sum(messageTable.table))
