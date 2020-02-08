# %% [markdown]
# #

import time

import numpy as np
import matplotlib.pyplot as plt
from graspy.match import FastApproximateQAP
from graspy.plot import heatmap
from graspy.simulations import sbm


def invert_permutation(p):
    """The argument p is assumed to be some permutation of 0, 1, ..., len(p)-1. 
    Returns an array s, where s[i] gives the index of i in p.
    """
    s = np.empty(p.size, p.dtype)
    s[p] = np.arange(p.size)
    return s


n = [50, 20, 20, 5, 5]
block_p = np.zeros((5, 5))
block_p += np.diag(0.5 * np.ones(5))


n_verts = 100

shuffle_inds = np.random.permutation(n_verts)
A = sbm(n, block_p)
B = A[np.ix_(shuffle_inds, shuffle_inds)]  # B is a permuted version of A (corr = 1)

faq = FastApproximateQAP(
    max_iter=30,
    eps=0.0001,
    init_method="rand",
    n_init=100,
    shuffle_input=False,
    maximize=True,
)

A_found, B_found = faq.fit_predict(A, B)

reverse_shuffle = invert_permutation(shuffle_inds)
heatmap(A - B_found)  # predicted permutation
heatmap(A - B[np.ix_(reverse_shuffle, reverse_shuffle)])  # optimal permutation
plt.show()

# %% [markdown]
# # Try NMI thingy
from sklearn.manifold import MDS
from sklearn.metrics import normalized_mutual_info_score
from graspy.utils import symmetrize

n_init = 10
faq = FastApproximateQAP(max_iter=30, eps=0.0001, init_method="rand", n_init=1)

found_perms = []
scores = []
for i in range(n_init):
    found_perms.append(faq.fit_predict(A, B))
    scores.append(faq.score_)

nmi_mat = np.zeros((n_init, n_init))
for i in range(n_init):
    for j in range(i + 1, n_init):
        nmi = normalized_mutual_info_score(found_perms[i], found_perms[j])
        nmi_mat[i, j] = nmi
        print(nmi)

nmi_mat = nmi_mat + nmi_mat.T
nmi_mat += np.diag(np.ones(n_init))
heatmap(nmi_mat)
