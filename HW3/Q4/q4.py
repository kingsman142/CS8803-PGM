import numpy as np
from scipy.io import loadmat

data_file = loadmat("SymptomDisease.mat")
W = data_file["W"] # 200x50 double
b = data_file["b"] # 200x1 double
p = data_file["p"] # 50x1 double
s = data_file["s"] # 200x1 logical

BURNIN = 10000
SUBSAMPLING = 1000
NUM_DISEASES = len(p)
NUM_SYMPTOMS = len(W)

def calc_likelihood(symptom_weights, symptom_bias, diseases, disease_priors, symptom_presence):
    likelihood = 1.0
    for s in range(NUM_SYMPTOMS):
        symptom_conditional = np.sigmoid( np.dot(symptom_weights[s], diseases) + symptom_bias[s] )
        if not symptom_presence[s]:
            symptom_conditional = 1 - symptom_conditional
        likelihood *= symptom_conditional
    for d in range(NUM_DISEASES):
        likelihood *= disease_priors[d]

    return likelihood

curr_probs = [0] * NUM_DISEASES # init value
estimates = []

samples_z0 = [samples[i] for i in range(N) if z[i] == 0] # samples from the first Gaussian component
samples_z1 = [samples[i] for i in range(N) if z[i] == 1] # samples from the second Gaussian component
samples_z0_mean, samples_z1_mean = np.mean(samples_z0), np.mean(samples_z1) # individual sample means
posterior_std = 100 / (101*N) # proof of this is shown in the report
mu0_posterior_mean, mu1_posterior_mean = (100*samples_z0_mean)/101, (100*samples_z1_mean)/101 # proof of this is shown in the report

indices_to_choose_from = [i for i in range(NUM_DISEASES)]
for i in range(BURNIN + SUBSAMLING):
    random_index = np.random.choice(indices_to_choose_from, size = 1)
    new_val = np.random.normal(mu0_posterior_mean, posterior_std)
    curr_probs[random_index] = new_val

    if i > BURNIN: # only keep information from last SUBSAMPLING samples
        estimates.append(list(curr_probs))
final_mu = np.mean(estimates, axis = 0) # average means across the last SUBSAMPLING samples
print("Gibbs estimated means: {}".format(final_mu))
