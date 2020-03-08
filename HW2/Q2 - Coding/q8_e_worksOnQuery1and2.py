import numpy as np
from scipy.io import loadmat

class Factor():
    def __init__(self, table, condVars, inputVars, givenVars, symptomsInit = False, diseasesInit = False):
        self.table = table
        self.condVars = condVars
        self.inputVars = inputVars
        self.allVars = self.condVars + self.inputVars

        diseaseIndices = [1, 2, 3, 4]
        self.diseasesLeft = [index for index in diseaseIndices if not index in givenVars]

        self.convert_matlab_table()

        #print(self.condVars, self.inputVars, self.diseasesLeft)

        if symptomsInit:
            # only keep the dimensions that are not given
            if diseaseIndices[0] in givenVars:
                self.table = self.table[1]
                self.condVars.remove(diseaseIndices[0])
            if diseaseIndices[1] in givenVars:
                self.table = self.table[:, 1]
                self.condVars.remove(diseaseIndices[1])
            if diseaseIndices[2] in givenVars:
                self.table = self.table[:, :, 1]
                self.condVars.remove(diseaseIndices[2])
            if diseaseIndices[3] in givenVars:
                self.table = self.table[:, :, :, 1]
                self.condVars.remove(diseaseIndices[3])
            if inputVars[0] in givenVars:
                self.table = self.table[:, :, :, :, 1]
                self.inputVars.remove(inputVars[0])
        elif diseasesInit:
            # only keep the dimensions that are not given
            if 0 in givenVars:
                self.table = self.table[1]
                self.condVars.remove(0)
            if inputVars[0] in givenVars:
                self.table = self.table[:, 1]
                self.inputVars.remove(inputVars[0])
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
            #print("diseases left:", self.diseasesLeft, len(self.diseasesLeft))
            newVars += self.allVars if not len(self.table.shape) == len(self.diseasesLeft) else []
            newVars += newFactor.allVars if not len(newTable.shape) == len(self.diseasesLeft) else []
            #print("newVars:", newVars, self.inputVars, newFactor.inputVars, len(self.table.shape), len(newTable.shape), len(self.diseasesLeft))
        newVars = list(set(newVars))
        #print(newTable.shape)
        #print(self.table.shape)
        #print(newMessageTable.shape)

        for i in range(0, int(2**len(newVars))):
            assignments = toBinaryArray(i, len(newVars)) #len(self.diseasesLeft)) # currently storing assignments for x1, x2, x3, x4, and x10
            diseaseAssignments = assignments[0:len(self.diseasesLeft)] # currently storing assignments for x1, x2, x3, x4

            currTable = newTable
            for assignment in diseaseAssignments: # traverse the table to get this combination's assignment
                #print(assignment)
                currTable = currTable[assignment] # newTable[x1][x2][x3][x4]
            newFalseTrueVars = []
            if len(newTable.shape) == len(self.diseasesLeft): # this table is a table with a given variable (e.g. x8), so it doesn't have a true and false column, just a true column (its shape is 2x2x2x2, not 2x2x2x2x2)
                #print(currTable)
                newFalseTrueVars.append(currTable)
                newFalseTrueVars.append(currTable)
            else:
                newFalseTrueVars.append(currTable[0]) #newVarFalse = currTable[0] # x10 = false
                newFalseTrueVars.append(currTable[1]) #newVarTrue = currTable[1] # x10 = true

            currNewMessageTable = newMessageTable
            for assignment in assignments[:-2]:
                currNewMessageTable = currNewMessageTable[assignment]
            #print(currNewMessageTable)
            if len(self.table.shape) == len(self.diseasesLeft):
                #print("{} currNewMessageTable[{}] ({}) *= {}".format(assignments, assignments[-1], currNewMessageTable[assignments[-1]], newFalseTrueVars[0]))
                currNewMessageTable = currNewMessageTable[assignments[-2]]
                currNewMessageTable[assignments[-1]] *= newFalseTrueVars[0]
            else:
                # TODO: control for the case when we have a 2x2x2x2x1 and a 2x2x2x2x2 table
                if False: #len(self.table.shape) == len(self.diseasesLeft):
                    pass
                else: # multiplying a 2x2x2x2x2 table by a 2x2x2x2x2 table
                    #currNewMessageTable = currNewMessageTable[assignments[-1]]
                    #print(i, currNewMessageTable.shape)
                    for val in range(0, len(currNewMessageTable)): # is it 1 or 2?
                        currNewMessageTable[val][0] *= newFalseTrueVars[0]
                        currNewMessageTable[val][1] *= newFalseTrueVars[1]
                                #currNewMessageTable[0] *= newFalseTrueVars[0] # 0000 0
                                #currNewMessageTable[1] *= newFalseTrueVars[1] # 0000 1 -- could be currNewMessageTable[0] as well, it's kind of arbitrary, since they have the same value

                #currNewMessageTable[0][0] *= newFalseTrueVars[0] # 0000 0
                #currNewMessageTable[1][0] *= newFalseTrueVars[0] # 0000 1 -- could be currNewMessageTable[0] as well, it's kind of arbitrary, since they have the same value
                #currNewMessageTable[0][1] *= newFalseTrueVars[1]
                #currNewMessageTable[1][1] *= newFalseTrueVars[1]

        # handle some logistics
        self.condVars = newVars
        self.inputVars = []
        self.allVars = newVars
        self.table = newMessageTable.squeeze()
        #print(self.allVars)
        #print(self.table.shape, len(self.table.shape))
        assert(len(self.allVars) == len(self.table.shape))

    def mult_factor_disease(self, newFactor, marginalizedVar):
        #print("HERE")
        newVars = set(self.get_vars()).union(set(newFactor.get_vars())) # combine [0, 4] and [1, 2, 3, 4] into set(0, 1, 2, 3)
        newVars.remove(marginalizedVar)
        newVars = sorted(list(newVars)) # convert from set(0, 1, 2, 3) to [0, 1, 2, 3]

        #print(self.allVars, newFactor.get_vars(), newVars, marginalizedVar)

        #print("MARGINALIZING OVER {}".format(marginalizedVar))
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

            #print("false assignment: ", marginalizedVarFalseAssignment)
            #print("true assignment: ", marginalizedVarTrueAssignment)
            newFactorFalse = newFactor.getProb(marginalizedVarFalseAssignment) # disease table probability when marginalized variable is false
            newFactorTrue = newFactor.getProb(marginalizedVarTrueAssignment) # disease table prob when marginalized variable is true
            thisFalse = self.getProb(marginalizedVarFalseAssignment) # messageTable probability when marginalized variable is false
            thisTrue = self.getProb(marginalizedVarTrueAssignment) # messageTable probability when marginalized variable is true

            #print(newFactorFalse, thisFalse, newFactorTrue, thisTrue)
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
        newVars.remove(marginalizedVar)
        newVars = sorted(list(newVars))

        #print("COMBINING {} FACTORS AND MARGINALIZING OVER NODE {}".format(len(newFactors), marginalizedVar))
        #print("NEW VARS: {}".format(newVars))

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
            #print(marginalizedVarFalseAssignment)
            #print(marginalizedVarTrueAssignment)

            newProbFalse = 1.0
            newProbTrue = 1.0
            for factor in newFactors:
                factorFalse = factor.getProb(marginalizedVarFalseAssignment) # disease table probability when marginalized variable is false
                factorTrue = factor.getProb(marginalizedVarTrueAssignment) # disease table prob when marginalized variable is true
                #print("here2", factorFalse, factorTrue)

                newProbFalse *= factorFalse
                newProbTrue *= factorTrue

            newProb = newProbFalse + newProbTrue #newFactorFalse * thisFalse + newFactorTrue * thisTrue # sum over the marginalized variable, and for each summation, there is one multiplication

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
        assignments = []
        #print("here")
        for var in self.allVars: # if vars = [1, 2, 3, 4], then map varAssignments {4: 1, 0: 1, 2: 0, 3: 1, 1: 0} to [0, 0, 1, 1]
            #print(var, varAssignments[var], varAssignments, self.allVars)
            assignments.append(varAssignments[var])

        # iterate over the table and find the probability that matches the corresponding variable assignments
        currTable = self.table
        for assignment in assignments:
            currTable = currTable[assignment]

        #print(self.allVars, assignments, currTable)

        return currTable

    def convert_matlab_table(self):
        newTable = np.zeros((2, ) * (len(self.allVars))) # convert the table from 16x2 to 2x2x2x2x2
        #print(self.table.shape)
        #print(newTable.shape)
        if len(self.allVars) == 1: # special case: isSummer table
            #print("\tCONVERT MATLAB TABLE SPECIAL CASE")
            newTable[0] = self.table[0]
            newTable[1] = self.table[1]
            self.table = newTable
            return

        #print("\tCONSTRUCTING TABLE... SIZE {}, SHAPE {}".format(newTable.size, newTable.shape))
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
which_query = 3
if which_query == 1:
    givens = [8, 11]
    queries = [1]
elif which_query == 2:
    givens = [4]
    queries = [7, 8, 9, 10, 11]
elif which_query == 3:
    givens = [0]
    queries = [10]
diseasesIndices = [1, 2, 3, 4]
for g in givens:
    assignments[g] = 1
data = loadmat("q8_vars.mat")
symptoms = data["symptoms"]
diseases = data["diseases"]
summerTable = data["summer_probs"]
#print(summerTable.shape)
#print(summerTable)
#print(diseases.shape)
#print(diseases)

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

    #print("diseases indices: ", diseasesIndices)
    newFactor = Factor(symptoms[i-5], condVars = list(diseasesIndices), inputVars = [i], givenVars = givens, symptomsInit = True)
    #newFactor.print_table()

    if messageTable is None: # initialize messageTable
        messageTable = newFactor
        #variablesLeft.append(i)
        continue # go to next iteration/factor

    messageTable.mult_factor_symptom(newFactor) # combineSymptomsTables(messageTable, newTable, diseasesLeft)
    #messageTableVars.append(i) # add this symptom to the table list

#print(messageTable.get_vars())
#messageTable.print_table()

# combine the diseases tables
for i in range(4, 0, -1):
    newFactor = Factor(diseases[i-1], condVars = [0], inputVars = [i], givenVars = givens, diseasesInit = True)
    #print("PRINTING DISEASE {}".format(i))
    #print(diseases[i-1])
    #newFactor.print_table()
    if i in queries + givens:
        remainingFactors.append(newFactor)
        continue

    if messageTable is None: # in the off chance we marginalized away all the symptoms, then the message table will begin as a disease table
        messageTable = newFactor
        continue

    messageTable.mult_factor_disease(newFactor, i)

#print(messageTable.get_vars())

# combine the isSummer node
for i in [0]:
    newFactor = Factor(summerTable, condVars = [0], inputVars = [], givenVars = givens)
    remainingFactors.append(newFactor)

    if messageTable is None:
        messageTable = newFactor # really weird scenario where no variables are given, so they're all marginalized out and all we get is P(x0)
        continue

#print(len(remainingFactors))

remainingFactors.append(messageTable)
messageTable.mult_factor_multiple(remainingFactors, 0) # marginalize over 0

messageTable.normalize_table()
#print(messageTable.get_table())
messageTable.print_table()
print(messageTable.get_vars())
print("Sum:", np.sum(messageTable.table))

#fluFalse = messageTable.table[0] / (messageTable.table[0] + messageTable.table[1])
#fluTrue = messageTable.table[1] / (messageTable.table[0] + messageTable.table[1])

#print("flu == FALSE : {}".format(fluFalse))
#print("flu == TRUE : {}".format(fluTrue))
