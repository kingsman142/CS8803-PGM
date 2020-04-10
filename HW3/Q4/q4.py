import math
import numpy as np
from scipy.io import loadmat
from scipy.special import expit

data_file = loadmat("SymptomDisease.mat")
W = data_file["W"] # 200x50 double
b = data_file["b"] # 200x1 double
p = data_file["p"].flatten() # 50x1 double
s = data_file["s"] # 200x1 logical

BURNIN = 100
SUBSAMPLING_RATE = 5
SUBSAMPLING_ITERATIONS = 300
NUM_DISEASES = len(p)
NUM_SYMPTOMS = len(W)

print("BURNIN: {}".format(BURNIN))
print("SUBSAMPLING RATE: {}".format(SUBSAMPLING_RATE))
print("SUBSAMPLING ITERATIONS: {}".format(SUBSAMPLING_ITERATIONS))

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def get_random_sample_and_prob(symptoms, diseases, W, b, disease_index, disease_priors):
    # calculate two probabilities: one being if the disease is true, and the other being if the disease is false
    # NOTE: using logsum instead of multiplication
    false_x_prob = 0.0
    true_x_prob = 0.0
    false_disease = list(diseases)
    false_disease[disease_index] = 0
    true_disease = list(diseases)
    true_disease[disease_index] = 1

    for index, weight in enumerate(W):
        if symptoms[index][0] == 1:
            false_x_prob += np.log(sigmoid(np.dot(weight, false_disease) + b[index]))
            true_x_prob += np.log(sigmoid(np.dot(weight, true_disease) + b[index]))
        else: # symptoms[index][0] == 0
            false_x_prob += np.log(1.0 - sigmoid(np.dot(weight, false_disease) + b[index]))
            true_x_prob += np.log(1.0 - sigmoid(np.dot(weight, true_disease) + b[index]))
    for index, prior in enumerate(disease_priors.flatten()):
        if index == disease_index:
            false_x_prob += np.log(1.0 - prior)
            true_x_prob += np.log(prior)
        else:
            actual_prior = prior if diseases[index] == 1 else (1-prior) # is the disease true or false?
            false_x_prob += np.log(actual_prior)
            true_x_prob += np.log(actual_prior)

    # convert back from scores to probabilities
    false_x_prob = np.exp(false_x_prob)
    true_x_prob = np.exp(true_x_prob)

    # normalize true probability
    false_x_prob_normalized = false_x_prob / (false_x_prob + true_x_prob)
    true_x_prob_normalized = true_x_prob / (false_x_prob + true_x_prob)

    # which one is more likely? this disease being true or false? return tuple with most likely option and the probability of the disease being true
    return (1 if true_x_prob_normalized > false_x_prob_normalized else 0, true_x_prob_normalized[0])

curr_probs = [0] * NUM_DISEASES # init value
probs = [0 for i in range(NUM_DISEASES)]
counts = [0] * NUM_DISEASES

for i in range(BURNIN + SUBSAMPLING_ITERATIONS):
    if i == 0 or i % 50 == 0:
        print("Finished iteration {}/{}".format(i+1, BURNIN + SUBSAMPLING_ITERATIONS))
    random_index = int(np.random.choice(NUM_DISEASES, size = 1))#i % NUM_DISEASES
    new_val, _ = get_random_sample_and_prob(s, curr_probs, W, b, random_index, p)
    curr_probs[random_index] = new_val

    if i >= BURNIN and i % SUBSAMPLING_RATE == 0:
        for j in range(NUM_DISEASES):
            _, marginal = get_random_sample_and_prob(s, curr_probs, W, b, j, p)
            probs[j] += marginal
            counts[j] += 1

print(curr_probs)
probs = [round(probs[i] / counts[i], 4) for i in range(NUM_DISEASES)]
print(probs)
