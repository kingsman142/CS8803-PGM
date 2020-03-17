import numpy as np
import matplotlib.pyplot as plt

MU_UNICODE = "\u03BC"
SIGMA_UNICODE = "\u03C3"
SUBSCRIPT_0_UNICODE = "\u2080"
SUBSCRIPT_1_UNICODE = "\u2081"
THINSPACE_UNICODE = "\u200A"

ALGORITHM_ITERATIONS = 10000
ESTIMATES_ITERATIONS = 1000

# used for part a: generate N samples from the mixture of Gaussians distribution defined in the question
def draw_from_dist(N):
    samples = []
    z = [] # latent variable representing which Gaussian component the variable came from
    for n in range(N):
        if np.random.rand(1)[0] < 0.5:
            samples.append(np.random.normal(-5, 1))
            z.append(0)
        else:
            samples.append(np.random.normal(5, 1))
            z.append(1)
    return np.array(samples), z

# used for part b: draw from the proposal distribution for each mean's distribution (assume they're independent)
def draw_from_proposal_dist(mu_pair, sig):
    normal1 = np.random.normal(mu_pair[0], sig)
    normal2 = np.random.normal(mu_pair[1], sig)
    return (normal1, normal2)

def calc_acceptance(new_mu, old_mu, sigma, samples):
    acceptance_dist = lambda mu : np.exp(-0.5 * (mu[0]**2) / 100.0)*np.exp(-0.5 * (mu[1]**2) / 100.0) # prior probability
    likelihood = lambda mu, data : 0.5*np.exp(-0.5 * (data - mu[0])**2) + 0.5*np.exp(-0.5 * (data - mu[1])**2) # likelihood of data

    proposedmu_ratio = np.log(acceptance_dist(new_mu)) + np.sum(np.log(likelihood(new_mu, samples)))
    oldmu_ratio = np.log(acceptance_dist(old_mu)) + np.sum(np.log(likelihood(old_mu, samples)))
    ratio_diff = np.exp(proposedmu_ratio - oldmu_ratio)

    #return min(1, (acceptance_dist(new_mu)*np.prod(likelihood(new_mu, samples))) / (acceptance_dist(old_mu)*np.prod(likelihood(old_mu, samples))))
    return min(1, ratio_diff)

########################################
# PART A
########################################
N = 100
samples, z = draw_from_dist(N = N) # generate 100 samples
print("Distribution mean: {}\nDistribution std: {}\n".format(np.mean(samples), np.std(samples)))

# plot the distribution
'''num_bins = 10
plt.hist(samples, bins = num_bins)
plt.savefig("q6-{}samples-{}bins-distribution.png".format(N, num_bins))
plt.show()'''

########################################
# PART B (Metropolis-Hastings)
########################################
curr_mu = (0, 0) # init value
sigma = 5
estimates = []
num_acceptances = 0
for i in range(ALGORITHM_ITERATIONS + ESTIMATES_ITERATIONS):
    proposed_mu = draw_from_proposal_dist(curr_mu, sigma)
    acceptance_ratio = calc_acceptance(proposed_mu, curr_mu, sigma, samples)

    r = np.random.rand(1)[0]
    if r <= acceptance_ratio: # accept the candidate
        curr_mu = proposed_mu
        num_acceptances += 1

    if i >= ALGORITHM_ITERATIONS: # only keep information from last ESTIMATES_ITERATIONS samples
        estimates.append(list(curr_mu))
x = [min(pair[0], pair[1]) for pair in estimates] # estimates for mu0
y = [max(pair[0], pair[1]) for pair in estimates] # estimates for mu1
final_mu = (np.mean(x), np.mean(y)) # average means across the last ESTIMATES_ITERATIONS samples
print("Metropolis-Hastings estimated means: {}".format(final_mu))
print("Acceptance rate: {}\n".format(num_acceptances / (ALGORITHM_ITERATIONS + ESTIMATES_ITERATIONS))) # acceptance rate = (number of iterations we accepted) / (total # of iterations)

# plot the (x, y) coordinate pairs representing (mu0, mu1)
plt.title("MH estimated means ({}{}, {}{}) with {} = {}".format(MU_UNICODE, SUBSCRIPT_0_UNICODE, MU_UNICODE, SUBSCRIPT_1_UNICODE, SIGMA_UNICODE, sigma))
plt.xlabel("{}{}{}".format(MU_UNICODE, THINSPACE_UNICODE, SUBSCRIPT_0_UNICODE))
plt.ylabel("{}{}{}".format(MU_UNICODE, THINSPACE_UNICODE, SUBSCRIPT_1_UNICODE))
plt.scatter(x, y)
plt.savefig("mh-estimated-means-with-sigma-{}.png".format(sigma))

########################################
# PART C (Gibbs Sampling)
########################################
curr_mu = [0, 0] # init value
estimates = []
samples_z0 = [samples[i] for i in range(N) if z[i] == 0] # samples from the first Gaussian component
samples_z1 = [samples[i] for i in range(N) if z[i] == 1] # samples from the second Gaussian component
samples_z0_mean, samples_z1_mean = np.mean(samples_z0), np.mean(samples_z1) # individual sample means

posterior_std = 100 / (101*N) # proof of this is shown in the report
mu0_posterior_mean, mu1_posterior_mean = (100*samples_z0_mean)/101, (100*samples_z1_mean)/101 # proof of this is shown in the report
for i in range(ALGORITHM_ITERATIONS + ESTIMATES_ITERATIONS):
    new_val = np.random.normal(mu0_posterior_mean, posterior_std)
    curr_mu[0] = new_val

    new_val = np.random.normal(mu1_posterior_mean, posterior_std)
    curr_mu[1] = new_val

    if i > ALGORITHM_ITERATIONS: # only keep information from last ESTIMATES_ITERATIONS samples
        estimates.append(list(curr_mu))
x = [pair[0] for pair in estimates] # estimates for mu0
y = [pair[1] for pair in estimates] # estimates for mu1
final_mu = (np.mean(x), np.mean(y)) # average means across the last ESTIMATES_ITERATIONS samples
print("Gibbs estimated means: {}".format(final_mu))

# plot the (x, y) coordinate pairs representing (mu0, mu1)
plt.figure()
plt.title("Gibbs estimated means ({}{}, {}{})".format(MU_UNICODE, SUBSCRIPT_0_UNICODE, MU_UNICODE, SUBSCRIPT_1_UNICODE))
plt.xlabel("{}{}{}".format(MU_UNICODE, THINSPACE_UNICODE, SUBSCRIPT_0_UNICODE))
plt.ylabel("{}{}{}".format(MU_UNICODE, THINSPACE_UNICODE, SUBSCRIPT_1_UNICODE))
plt.scatter(x, y)
plt.savefig("gibbs-estimated-means.png")
plt.show()
