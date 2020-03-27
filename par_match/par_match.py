# %% [markdown]
# ##

import random
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from joblib import Parallel, delayed

from graspy.match import GraphMatch as GMP
from graspy.plot import heatmap
from graspy.simulations import er_corr, sbm, sbm_corr


# setup
np.random.seed(8888)
directed = False
loops = False
n_per_block = 150
n_blocks = 3
block_members = np.array(n_blocks * [n_per_block])
n_verts = block_members.sum()
rho = 0.5
block_probs = np.array([[0.7, 0.3, 0.4], [0.3, 0.7, 0.3], [0.4, 0.3, 0.7]])

# plot block probs
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
sns.heatmap(block_probs, cbar=False, annot=True, square=True, cmap="Reds", ax=ax)
ax.set_title("SBM block probabilities")

# generate graphs
A1, A2 = sbm_corr(block_members, block_probs, rho, directed=directed, loops=loops)
fig, axs = plt.subplots(1, 3, figsize=(10, 5))
heatmap(A1, ax=axs[0], cbar=False, title="Graph 1")
heatmap(A2, ax=axs[1], cbar=False, title="Graph 2")
heatmap(A1 - A2, ax=axs[2], cbar=False, title="Diff (G1 - G2)")

# shuffle for testing
node_shuffle_input = np.random.permutation(n_verts)
A2_shuffle = A2[np.ix_(node_shuffle_input, node_shuffle_input)]
node_unshuffle_input = np.array(range(n_verts))
node_unshuffle_input[node_shuffle_input] = np.array(range(n_verts))

# plot shuffled
fig, axs = plt.subplots(1, 3, figsize=(10, 5))
heatmap(A1, ax=axs[0], cbar=False, title="Graph 1")
heatmap(A2_shuffle, ax=axs[1], cbar=False, title="Graph 2 shuffled")
heatmap(A1 - A2_shuffle, ax=axs[2], cbar=False, title="Diff (G1 - G2 shuffled)")


n_init = 100  # parameter for GMP

# run GMP in serial
currtime = time.time()

sgm = GMP(n_init=n_init, init_method="rand")
sgm = sgm.fit(A1, A2_shuffle)

A2_unshuffle = A2_shuffle[np.ix_(sgm.perm_inds_, sgm.perm_inds_)]

fig, axs = plt.subplots(1, 3, figsize=(10, 5))
heatmap(A1, ax=axs[0], cbar=False, title="Graph 1")
heatmap(A2_unshuffle, ax=axs[1], cbar=False, title="Graph 2 unshuffled")
heatmap(A1 - A2_unshuffle, ax=axs[2], cbar=False, title="Diff (G1 - G2 unshuffled)")

match_ratio = 1 - (
    np.count_nonzero(abs(sgm.perm_inds_ - node_unshuffle_input)) / n_verts
)
print("Match Ratio (serial) ", match_ratio)
print("Optimal objective (serial) ", sgm.score_)

print(f"{time.time() - currtime} elapsed for serial")

# run GMP in parallel
currtime = time.time()

seeds = np.random.choice(int(1e8), n_init, replace=False)


def run_gmp(seed):
    np.random.seed(seed)
    sgm = GMP(n_init=1, init_method="rand")
    sgm.fit(A1, A2_shuffle)
    return sgm.score_, sgm.perm_inds_


outs = Parallel(n_jobs=-1)(delayed(run_gmp)(seed) for seed in seeds)

outs = list(zip(*outs))
scores = outs[0]
perms = outs[1]
max_ind = np.argmax(scores)
optimal_perm = perms[max_ind]

A2_unshuffle = A2_shuffle[np.ix_(optimal_perm, optimal_perm)]

fig, axs = plt.subplots(1, 3, figsize=(10, 5))
heatmap(A1, ax=axs[0], cbar=False, title="Graph 1")
heatmap(A2_unshuffle, ax=axs[1], cbar=False, title="Graph 2 unshuffled")
heatmap(A1 - A2_unshuffle, ax=axs[2], cbar=False, title="Diff (G1 - G2 unshuffled)")

match_ratio = 1 - (np.count_nonzero(abs(optimal_perm - optimal_perm)) / n_verts)
print("Match Ratio (parallel): ", match_ratio)
print("Optimal objective (parallel) ", scores[max_ind])

print(f"{time.time() - currtime} elapsed for parallel")
