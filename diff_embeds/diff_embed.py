import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns 
from graspy.simulations import sbm
from graspy.plot import heatmap
from src.utils import n_to_labels

#2 methods for k
const_k = 2
def linear_k(slope, n):
    k = slope * n
    k = int(k)
    return k

#2 methods for q
const_q = 0.5
def decay_q(slope, n):
    q = slope / n
    return q

#Static Variables
slope_q = 100
slope_k = 1 / slope_q
p = 0.3
n_verts = [200, 300, 400, 500]
n_sims = 3

#Generate B Matrix
def B_matrix(k, p, q):
    B = np.zeros((k,k))
    np.fill_diagonal(B, p)
    B[B == 0] = q
    return B

#Generate graph
for n in n_verts:
    for _ in range(n_sims):
        k = linear_k(slope_k, n)
        q = decay_q(slope_q, n)
        B = B_matrix(k, p, q)
        cn = [n // k] * k
        node_labels = n_to_labels(cn).astype(int)
        G = sbm(cn, B)
        heatmap(G, title=f"k={k} q={q}", inner_hier_labels=node_labels)
        plt.show()