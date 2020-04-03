import math
import numpy as np
from scipy.io import loadmat
from scipy.special import expit
from scipy.special import kl_div

def valid_xy_coords(pair):
    return pair[0] in range(0, 3) and pair[1] in range(0, 3)
def valid_z_coords(coord):
    return coord in range(0, 3)

data_file = loadmat("p.mat")
p = data_file["p"][0][0] # contains 2 variables, "variables" and "table"
variables = p["variables"][0] # [1, 2, 3]
table = p["table"] # 3x3x3 matrix

# init Q
q_xy = np.random.uniform(low = 0.0, high = 1.0, size = (3, 3)) #np.random.normal(loc = [0.5], scale = [0.5], size = (3, 3))
q_z = np.random.uniform(low = 0.0, high = 1.0, size = (3)) #np.random.normal(loc = 0.5, scale = 0.5, size = 3)

# normalize distributions so they sum to 1
q_xy /= np.sum(q_xy)
q_z /= np.sum(q_z)

print(q_xy)
print(q_z)

# carry out mean-field iterative algorithm
NUM_ITERATIONS = 1000
EPSILON = 1e-4
counter = 0
unprocessed_nodes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
while True: # XY distribution
    # which variable will we modify during this iteration?
    index = counter % 9
    if not index in unprocessed_nodes: # skip this iteration
        counter += 1
        continue
    x = index % 3
    y = int(index / 3)
    old_prob = q_xy[x][y]

    sum = 0.0
    count = 0

    first = [x-1, y]
    second = [x+1, y]
    third = [x, y-1]
    fourth = [x, y+1]
    new_coords = [first, second, third, fourth]
    new_unprocessed_nodes = []
    for coords in new_coords:
        if valid_xy_coords(coords): # valid neighbor coordinates
            sum += q_xy[coords[0]][coords[1]]
            count += 1
            new_unprocessed_nodes.append(coords[1]*3 + coords[0]) # 3y + x

    new_prob = sum / count # compute average probability of neighbors
    q_xy[x][y] = new_prob
    q_xy /= np.sum(q_xy) # renormalize distribution

    #print(abs(new_prob - old_prob))
    # the probability of those node changed, so change its neighbor nodes to unprocessed
    if abs(new_prob - old_prob) > EPSILON:
        for node in new_unprocessed_nodes: # add all the neighbors of this node to the unprocessed nodes list
            if not node in unprocessed_nodes:
                unprocessed_nodes.append(node)
    else:
        unprocessed_nodes.remove(index)

    # we don't have anymore nodes to process, so we have finally converged on a distribution
    if not len(unprocessed_nodes):
        break

    # update counter
    counter += 1

# Z distribution
counter = 0
unprocessed_nodes = [0, 1, 2]
while True:
    # which variable will we modify during this iteration?
    index = counter % 3
    if not index in unprocessed_nodes: # skip this iteration
        counter += 1
        continue
    old_prob = q_z[index]

    sum = 0.0
    count = 0

    first = index-1
    second = index+1
    new_coords = [first, second]
    new_unprocessed_nodes = []
    for coords in new_coords:
        if valid_z_coords(coords): # valid neighbor coordinates
            sum += q_z[coords]
            count += 1
            new_unprocessed_nodes.append(coords)

    new_prob = sum / count # compute average probability of neighbors
    q_z[index] = new_prob
    q_z /= np.sum(q_z) # renormalize distribution

    #print(abs(new_prob - old_prob))
    # the probability of those node changed, so change its neighbor nodes to unprocessed
    if abs(new_prob - old_prob) > EPSILON:
        for node in new_unprocessed_nodes: # add all the neighbors of this node to the unprocessed nodes list
            if not node in unprocessed_nodes:
                unprocessed_nodes.append(node)
    else:
        unprocessed_nodes.remove(index)

    # we don't have anymore nodes to process, so we have finally converged on a distribution
    if not len(unprocessed_nodes):
        break

    # update counter
    counter += 1

print(q_xy)
print(q_z)

print(q_xy.shape)
print(q_z.shape)
q = np.outer(q_xy, q_z)
q = np.reshape(q, (3, 3, 3))
q /= np.sum(q)
print(q)
print("KL Divergence: {}".format(np.sum(kl_div(table, q))))
