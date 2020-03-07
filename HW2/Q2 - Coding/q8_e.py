import numpy as np

class Factor():
    def __init__(self, table, condVars, inputVars, givenVars, symptomsInit = False):
        self.table = table
        self.condVars = condVars
        self.inputVars = inputVars
        self.allVars = self.condVars + self.inputVars

        self.convert_matlab_table() # TODO: this should only be called when the initial raw matlab table is passed in
        variableColumns = self.condVars + self.inputVars # each index of the table will correspond to a variable (e.g. x1, x2, x3, x4, x8 correspond to table.shape[0], table.shape[1], ... respectively)

        # TODO: create a list to keep track of which variables are in this factor after marginalizing below
        if symptomsInit:
            diseaseIndices = [1, 2, 3, 4] # TODO: make this not hardcoded
            # only keep the dimensions that are given
            if diseaseIndices[0] in givenVars:
                newTable = newTable[1]
            if diseaseIndices[1] in givenVars:
                newTable = newTable[:, 1]
            if diseaseIndices[2] in givenVars:
                newTable = newTable[:, :, 1]
            if diseaseIndices[3] in givenVars:
                newTable = newTable[:, :, :, 1]
            if inputVars[0] in givenVars:
                newTable = newTable[:, :, :, :, 1]

    def mult_factor_symptom(newFactor):
        messageTable = self.table # TODO: substitute below messageTable for self.table
        newTable = newFactor.get_table() # TODO: substitute out vars below for newFactor.get_table()
        diseasesLeft = [1, 2, 3, 4] # TODO: this is hardcoded in, add in something in the constructor

        # TODO: create a newVars variable
        #newMessageTable = np.zeros(np.prod(messageTable.shape)*2, 1) # TODO -- maybe replace the below line with this line?
        # TODO: control for the case where the input table is 16 elements rather than 32 elements because the value is given (for example, in query 1)
        # TODO: (kind of related to above) if both tables are just 2x2x2x1, then the result should be 2x2x2
        newMessageTable = np.stack((messageTable, messageTable), axis = -1) # converts from 2x2x2x2x2 to 2x2x2x2x2x2 (1 more dimension)
        for i in range(0, int(2**len(diseasesLeft))):
            assignments = toBinaryArray(i) # currently storing assignments for x1, x2, x3, x4, and x10
            diseaseAssignments = assignments[0:len(diseasesLeft)] # currently storing assignments for x1, x2, x3, x4

            currTable = newTable
            for assignment in diseaseAssignments: # traverse the table to get this combination's assignment
                currTable = currTable[assignment]
            newVarFalse = currTable[0]
            newVarTrue = currTable[1]

            currNewMessageTable = newMessageTable
            for assignment in assignments:
                currNewMessageTable = currNewMessageTable[assignment]
            currNewMessageTable[0] *= newVarFalse # 0000 0
            currNewMessageTable[1] *= newVarTrue # 0000 1 -- could be currNewMessageTable[0] as well, it's kind of arbitrary, since they have the same value

        # handle some logistics
        self.condVars = newVars
        self.inputVars = []
        self.allVars = newVars
        self.table = newMessageTable

    def mult_factor_disease(newFactor, marginalizedVar):
        newVars = set(self.get_vars()).union(set(newFactor.get_vars())) # combine [0, 4] and [1, 2, 3, 4] into set(0, 1, 2, 3)
        newVars.remove(marginalizedVar)
        newVars = sorted(list(newVars)) # convert from set(0, 1, 2, 3) to [0, 1, 2, 3]

        newFactorTable = self.zeros((2, ) * len(newVars))
        for i in range(0, 2^(len(newVars))): # from 0 to 7 inclusive
            assignments = toBinaryArray(i) # each variable in this assignment corresponds to the respective variable in newVars
            varAssignments = {}
            for index, assignment in enumerate(assignments):
                varAssignments[newVars[index]] = assignment # place all variable assignments into a collective dict

            marginalizedVarFalseAssignment = dict(varAssignments) # create variable assignment for when the marginalized variable is false
            marginalizedVarTrueAssignment = dict(varAssignments) # create variable assignment for when the marginalized variable is true
            marginalizedVarFalse[marginalizedVar] = 0
            marginalizedVarTrue[marginalizedVar] = 1

            newFactorFalse = newFactor.getProb(marginalizedVarFalseAssignment) # disease table probability when marginalized variable is false
            newFactorTrue = newFactor.getProb(marginalizedVarTrueAssignment) # disease table prob when marginalized variable is true
            thisFalse = self.getProb(marginalizedVarFalseAssignment) # messageTable probability when marginalized variable is false
            thisTrue = self.getProb(marginalizedVarTrueAssignment) # messageTable probability when marginalized variable is true

            newProb = newFactorFalse * thisFalse + newFactorTrue * thisTrue # sum over the marginalized variable, and for each summation, there is one multiplication

            # traverse the new factor table and insert this newly calculated probability
            currTable = newFactorTable
            for assignment in assignments[::-1]:
                currTable = currTable[assignment]
            currTable[assignments[-1]] = newProb

        # handle some higher-level logistics
        self.condVars = newVars
        self.inputVars = []
        self.allVars = newVars
        self.table = newFactorTable

    def mult_factor_multiple(newFactors, marginalizedVar):
        newVars = set(newFactors[0].get_vars())
        for factor in newFactors:
            newVars = newVars.union(factor.get_vars())
        newVars.remove(marginalizedVar)
        newVars = sorted(list(newVars))

        newFactorTable = self.zeros((2, ) * len(newVars))
        for i in range(2^len(newVars)):
            assignments = toBinaryArray(i)
            varAssignments = {}
            for index, assignment in enumerate(assignments):
                varAssignments[newVars[index]] = assignment # place all variable assignments into a collective dict

            marginalizedVarFalseAssignment = dict(varAssignments) # create variable assignment for when the marginalized variable is false
            marginalizedVarTrueAssignment = dict(varAssignments) # create variable assignment for when the marginalized variable is true
            marginalizedVarFalse[marginalizedVar] = 0
            marginalizedVarTrue[marginalizedVar] = 1

            newProbFalse = 1.0
            newProbTrue = 1.0
            for factor in newFactors:
                factorFalse = newFactor.getProb(marginalizedVarFalseAssignment) # disease table probability when marginalized variable is false
                factorTrue = newFactor.getProb(marginalizedVarTrueAssignment) # disease table prob when marginalized variable is true

                newProbFalse *= factorFalse
                newProbTrue *= factorTrue

            newProb = newProbFalse + newProbTrue #newFactorFalse * thisFalse + newFactorTrue * thisTrue # sum over the marginalized variable, and for each summation, there is one multiplication

            # traverse the new factor table and insert this newly calculated probability
            currTable = newFactorTable
            for assignment in assignments[::-1]:
                currTable = currTable[assignment]
            currTable[assignments[-1]] = newProb

        # handle some higher-level logistics
        self.condVars = newVars
        self.inputVars = []
        self.allVars = newVars
        self.table = newFactorTable

    def getProb(self, varAssignments):
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
        for i in range(0, 2**np.prod(newTable.shape)): # loop through all the possible 16 states
            assignments = toBinaryArray(i) # convert from i = 5 to [0, 1, 1]

            currTable = newTable
            for assignment in assignments[:-1]: # traverse to currTable[0][1]
                currTable = currTable[assignment]
            currTable[-1] = self.table[i] # assign currTable[0][1][1] = self.table[5]
        self.table = newTable

    def get_table(self):
        return self.table

    def get_vars(self):
        return self.allVars

def toBinaryArray(number, length = None):
    if length is not None:
        return [int(ch) for ch in list(format(number, 'b').zfill(length)[::-1])] # all elements are ints e.g. [1, 0, 1, 0, 0, 0, 0, 0] for int(5) -- LSB is at index 0
    else:
        return [int(ch) for ch in list(format(number, 'b')[::-1])]

# TODO: read in the matlab data (i.e. tables)

assignments = np.zeros(12, 1)
givens = [8, 11]
queries = [1]
diseasesIndices = [1, 2, 3, 4]
for g in givens:
    assignments[g] = 1

'''newSymptoms = np.zeros(7, 16)
newDiseases = np.zeros(4, 2)'''

variablesLeft = [1, 2, 3, 4]
messageTable = None

'''variablesLeft = [variable for variable in variablesLeft if not variable in givens] # remove variablesLefts that are givens (for example, P(x11 | x4), we don't want to marginalize over 4)
diseasesLeft = list(variablesLeft)
messageTableVars = diseasesLeft'''

remainingFactors = []

# combine the symptoms tables
for i in range(5, 12): # 5 to 11 inclusive
    if not i in givens + queries: # this symptom will be marginalized out, so skip this step
        continue

    newFactor = Factor(symptoms[i], condVars = diseases, inputVars = [i], givens = givens, symptomsInit = True)

    if messageTable is None: # initialize messageTable
        messageTable = newFactor
        #variablesLeft.append(i)
        continue # go to next iteration/factor

    messageTable = messageTable.mult_factor_symptom(newFactor, i) # combineSymptomsTables(messageTable, newTable, diseasesLeft)
    #messageTableVars.append(i) # add this symptom to the table list

# combine the diseases tables
for i in range(4, 1, -1):
    newFactor = Factor(diseases[i], condVars = [0], inputVars = [i], givens = givens)
    if i in queries:
        remainingFactors.append(newFactor)
        continue

    if messageTable is None: # in the off chance we marginalized away all the symptoms, then the message table will begin as a disease table
        messageTable = newFactor
        continue

    messageTable = messageTable.mult_factor_disease(newFactor, i)

# combine the isSummer node
for i in [0]:
    newFactor = Factor(summerTable, condVars = [], inputVars = [0], givens = givens)
    remainingFactors.append(newFactor)

    if messageTable is None:
        messageTable = newFactor # really weird scenario where no variables are given, so they're all marginalized out and all we get is P(x0)
        continue

finalTable = Factor.mult_factor_multiple(remainingFactors, 0) # marginalize over 0

print(finalTable)

fluFalse = finalTable.table[0] / (finalTable.table[0] + finalTable.table[1])
fluTrue = finalTable.table[1] / (finalTable.table[0] + finalTable.table[1])

print("flu == FALSE : {}".format(fluFalse))
print("flu == TRUE : {}".format(fluTrue))
