import math
import numpy as np
from scipy.io import loadmat
from scipy.special import expit

data_file = loadmat("SymptomDisease.mat")
W = data_file["W"] # 200x50 double
b = data_file["b"] # 200x1 double
p = data_file["p"] # 50x1 double
s = data_file["s"] # 200x1 logical

BURNIN = 1000
SUBSAMPLING = 10000
NUM_DISEASES = len(p)
NUM_SYMPTOMS = len(W)

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

'''def calc_likelihood(symptom_weights, symptom_bias, diseases, disease_priors, symptom_presence):
    likelihood = 1.0
    for s in range(NUM_SYMPTOMS):
        symptom_conditional = np.sigmoid( np.dot(symptom_weights[s], diseases) + symptom_bias[s] )
        if not symptom_presence[s]:
            symptom_conditional = 1 - symptom_conditional
        likelihood *= symptom_conditional
    for d in range(NUM_DISEASES):
        likelihood *= disease_priors[d]

    return likelihood'''

def get_random_sample(symptoms, diseases, W, b, disease_index, disease_prior):
    # calculate two probabilities: one being if the disease is true, and the other being if the disease is false
    '''diseases[disease_index] = 0
    false_x_prob = (np.dot(new_W, diseases) + new_b)*(1-disease_prior)#expit(np.dot(new_W, false_disease) + new_b)# * (1.0 - disease_prior) # probability of this random disease being true
    diseases[disease_index] = 1
    true_x_prob = (np.dot(new_W, diseases) + new_b)*disease_prior#expit(np.dot(new_W, true_disease) + new_b)# * disease_prior # probability of this random disease being true
    if disease_index == 1:
        print(false_x_prob, true_x_prob, disease_prior)
        pass'''
    false_x_prob = 0.0
    true_x_prob = 0.0
    false_disease = list(diseases)
    false_disease[disease_index] = 0
    true_disease = list(diseases)
    true_disease[disease_index] = 1
    for index, weight in enumerate(W):
        false_x_prob += np.log( sigmoid(np.dot(weight, false_disease) + b[index]) )
        true_x_prob += np.log( sigmoid(np.dot(weight, true_disease) + b[index]) )
    #true_x_prob = np.exp(true_x_prob/len(W)) * (disease_prior)
    #if disease_index == 1:
    #    pass#print(true_x_prob)
    #return 1 if np.random.uniform() < true_x_prob else 0

    # OLD
    false_x_prob += np.log(1.0 - disease_prior)
    true_x_prob += np.log(disease_prior)
    return 1 if true_x_prob > false_x_prob else 0 # which one is more likely? this disease being true or false?

    #true_x_prob = np.exp(true_x_prob)
    #false_x_prob = np.exp(false_x_prob)
    #true_x_prob = true_x_prob / (true_x_prob + false_x_prob)
    #if true_x_prob > 0.01:
    #    print(true_x_prob)
    #return 1 if np.random.uniform() < true_x_prob else 0

def get_prob(symptoms, diseases, W, b, disease_index, disease_prior):
    # calculate two probabilities: one being if the disease is true, and the other being if the disease is false
    false_x_prob = 0.0
    true_x_prob = 0.0
    false_disease = list(diseases)
    false_disease[disease_index] = 0
    true_disease = list(diseases)
    true_disease[disease_index] = 1
    for index, weight in enumerate(W):
        false_x_prob += np.log( expit(np.dot(weight, false_disease) + b[index]) )
        true_x_prob += np.log( expit(np.dot(weight, true_disease) + b[index]) )
    false_x_prob = np.exp(false_x_prob) * (1.0 - disease_prior)
    true_x_prob = np.exp(true_x_prob) * (disease_prior)

    #false_x_prob += np.log(1.0 - disease_prior)
    #true_x_prob += np.log(disease_prior)

    return true_x_prob / (true_x_prob + false_x_prob) # which one is more likely? this disease being true or false?

curr_probs = [1] * NUM_DISEASES # init value
estimates = []
probs = [0 for i in range(NUM_DISEASES)]
counts = [0] * NUM_DISEASES

# math to calculate the new weights and biases for the modified/shifted sigmoid
new_W = np.zeros(NUM_DISEASES) # NOTE: same size of the W vector
new_b = 0.0 # NOTE: same size as the bias vector
for index, weight in enumerate(W):
    new_W += (weight if s[index][0] else -weight)
    new_b += b[index][0]
new_W /= len(W)
new_b /= len(W)

indices_to_choose_from = [i for i in range(NUM_DISEASES)]
for i in range(BURNIN + SUBSAMPLING):
    if i % 1000 == 0:
        print(i)
    random_index = int(np.random.choice(NUM_DISEASES, size = 1))#i % NUM_DISEASES #int(np.random.choice(indices_to_choose_from, size = 1))
    #* curr_probs = np.random.randint(low = 0, high = 2, size = NUM_DISEASES)
    #print(curr_probs)
    new_val = get_random_sample(s, curr_probs, W, b, random_index, p[random_index]) #np.random.normal(mu0_posterior_mean, posterior_std)
    curr_probs[random_index] = new_val #*
    #print(curr_probs)
    #print(random_index)
    if random_index == 0:
        pass#print(new_val)

    if i >= BURNIN: # only keep information from last SUBSAMPLING samples
        probs[random_index] += new_val
        counts[random_index] += 1
        #estimates.append(list(curr_probs))

print(curr_probs)
for i in range(NUM_DISEASES):
    probs[i] = get_prob(s, curr_probs, W, b, i, p[i])
#print(len(estimates))
#final_mu = np.mean(estimates, axis = 0) # average means across the last SUBSAMPLING samples
#print("Gibbs estimated probs: {}".format(final_mu))
#print(counts)
#print(probs)
#* probs = [probs[i] / (counts[i]) for i in range(NUM_DISEASES)] #[x / (SUBSAMPLING / NUM_DISEASES) for x in probs]
print(probs)
