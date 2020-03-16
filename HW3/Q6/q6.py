import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

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

def calc_acceptance(new_mu, old_mu, sigma, data):
    #print(new_mu, old_mu)
    acceptance_dist = lambda mu : np.exp(-0.5 * (mu[0]**2) / 100.0)*np.exp(-0.5 * (mu[1]**2) / 100.0) # prior probability
    pam = lambda mu, mu2 : ss.norm(mu[0], sigma).pdf(mu2)*ss.norm(mu[1], sigma).pdf(mu2) #np.random.normal(mu[0], sigma)*np.random.normal(mu[1], sigma)
    likelihood = lambda mu : 0.5*np.exp(-0.5 * (data - mu[0])**2) + 0.5*np.exp(-0.5 * (data - mu[1])**2) # likelihood of data
    #print(np.prod(likelihood(new_mu)), np.prod(likelihood(old_mu)))
    #print(len(likelihood(old_mu)))
    #return np.prod(likelihood(new_mu)) / np.prod(likelihood(old_mu))
    #return (acceptance_dist(new_mu)*pam(new_mu)) / (acceptance_dist(old_mu)*pam(old_mu))
    #return min(1, (acceptance_dist(new_mu)*np.prod(likelihood(new_mu))*pam(old_mu, new_mu[0])*pam(old_mu, new_mu[1])) / (acceptance_dist(old_mu)*np.prod(likelihood(old_mu))*pam(new_mu, old_mu[0])*pam(new_mu, old_mu[1])))
    return min(1, (acceptance_dist(new_mu)*np.prod(likelihood(new_mu))) / (acceptance_dist(old_mu)*np.prod(likelihood(old_mu))))
    #return acceptance_dist(new_mu) / acceptance_dist(old_mu)
    #return min(1, new_acceptance / old_acceptance)

    #firstlike = ss.norm(new_mu[0], sigma).pdf(old_mu[0])*ss.norm(new_mu[1], sigma).pdf(old_mu[1])
    #secondlike = ss.norm(old_mu[0], sigma).pdf(new_mu[0])*ss.norm(old_mu[1], sigma).pdf(new_mu[1])
    #return min(1, (firstlike * acceptance_dist(new_mu))/(secondlike * acceptance_dist(old_mu)))

####################
# PART A
####################
N = 100
num_bins = 10
samples, z = draw_from_dist(N = N) # generate 100 samples
print("Distribution mean: {}\nDistribution std: {}".format(np.mean(samples), np.std(samples)))

# plot the distribution
#plt.hist(samples, bins = num_bins)
#plt.savefig("q6-{}samples-{}bins-distribution.png".format(N, num_bins))
#plt.show()

#samples2 = np.random.normal(1, 0.5, size = 10000)*np.random.normal(-3, 0.5, size = 10000)
#plt.hist(samples2, bins = 100)
#plt.show()

#acceptance_dist = lambda mu : np.exp(-0.5 * (mu[0]**2) / 100.0)*np.exp(-0.5 * (mu[1]**2) / 100.0)
#i = [j/10 for j in range(-500, 500)]
#samples3 = []
#for item in i:
#    samples3.append(acceptance_dist((item, -0)))
#plt.hist(samples3, bins = 50)
#plt.show()

####################
# PART B
####################
curr_mu = (0, 0)
sigma = 5
iterations = 10000 #10000 + 1000
estimates = []
for i in range(iterations + 1000):
    proposed_mu = draw_from_proposal_dist(curr_mu, sigma)
    acceptance_ratio = calc_acceptance(proposed_mu, curr_mu, sigma, samples)
    #print(proposed_mu, curr_mu)
    #print(proposed_mu)
    #if i < 100:
    #    print(proposed_mu, curr_mu)
    #if i in range(10000, 10100):
    #    print(proposed_mu, acceptance_ratio)
    #print(acceptance_ratio)
    r = np.random.rand(1)[0]
    if r <= acceptance_ratio: # accept the candidate
        #print("ACCEPT CANDIDATE:", acceptance_ratio, r, i)
        curr_mu = proposed_mu
    if i >= iterations:
        estimates.append(curr_mu)
x = [min(pair[0], pair[1]) for pair in estimates]
y = [max(pair[0], pair[1]) for pair in estimates]
#plt.scatter(x, y)
#plt.show()
final_mu = (np.mean(x), np.mean(y))
print(curr_mu, final_mu, calc_acceptance(final_mu, curr_mu, sigma, samples))

####################
# PART C
####################
possible_indices = [0, 1] #[i for i in range(N)]
sigma = 0.5
curr_mu = [0, 0]
estimates = []
samples_z0 = [samples[i] for i in range(N) if z[i] == 0] # samples from the first Gaussian component
samples_z1 = [samples[i] for i in range(N) if z[i] == 1] # samples from the second Gaussian component
samples_z0_mean, samples_z1_mean = np.mean(samples_z0), np.mean(samples_z1)
print("means:", samples_z0_mean, samples_z1_mean)
for i in range(iterations + 10):
    component_index = np.random.choice(possible_indices, size = 1, replace = False)[0]
    #new_data = [samples[i] for i in range(N) if not i == component_index]

    draw = lambda mu : np.random.normal(mu, sigma) #0.5*np.random.normal(mu[0], 1) + 0.5*np.random.normal(mu[1], 1)
    draw2 = lambda mean, std : np.random.normal(mean, std)
    #draw2 = lambda mu : np.random.normal(mu[0], sigma)*np.random.normal(mu[1], sigma)

    #draw3 = lambda
    #draw = lambda mu : np.exp(-0.5 * (mu[0]**2) / 100.0)*np.exp(-0.5 * (mu[1]**2) / 100.0)

    #new_val, _ = draw_from_proposal_dist(curr_mu, sigma)
    new_val = draw2(100*samples_z0_mean/101, 100/(101*N)) #draw(curr_mu[0])#draw2(curr_mu)#draw(curr_mu[0])
    #print(new_val)
    curr_mu[0] = new_val #component_index] = new_val

    new_component_index = 1 if component_index == 0 else 0
    #new_val, _ = draw(curr_mu)#draw_from_proposal_dist(curr_mu, sigma)
    new_val = draw2(100*samples_z1_mean/101, 100/(101*N)) #draw(curr_mu[1])#draw2(curr_mu)#draw(curr_mu[1])
    curr_mu[1] = new_val #new_component_index] = new_val
    if i > iterations:
        estimates.append(curr_mu)
x = [pair[0] for pair in estimates]
y = [pair[1] for pair in estimates]
#plt.scatter(x, y)
#plt.show()
print(curr_mu, (np.mean(x), np.mean(y)))
