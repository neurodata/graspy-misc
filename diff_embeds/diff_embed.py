import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns 
import pandas as pd
from graspy.simulations import sbm
from graspy.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed
from graspy.plot import heatmap, pairplot
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from src.utils import n_to_labels

#2 methods for k
const_k = 2
def linear_k(slope, n):
    k = n / slope
    k = int(k)
    return k

#2 methods for q
const_q = 0.5
def decay_q(slope, n):
    q = slope / n
    return q

#Static Variables
slope = 50
p = 0.5
n_verts = [100, 150, 200, 250, 300]
n_sims = 3
embed = AdjacencySpectralEmbed()
temp = []
ari_vals = []

#Generate B Matrix
def B_matrix(k, p, q):
    B = np.zeros((k,k))
    np.fill_diagonal(B, p)
    B[B == 0] = q
    return B

#Generate graph
for n in n_verts:
    for _ in range(n_sims):
        k = linear_k(slope, n)
        q = decay_q(slope, n)
        B = B_matrix(k, p, q)
        cn = [n//k] * k
        labels_sbm = n_to_labels(cn).astype(int)
        G = sbm(cn, B)

        #embedding
        Xhat = embed.fit_transform(G)

        #clustering
        clust = KMeans(n_clusters = k)
        labels_clust = clust.fit_predict(Xhat)
        ari = adjusted_rand_score(labels_sbm, labels_clust)
        temp.append(ari)
    ari_vals.append(temp)
    temp = []     

for xe, ye in zip(n_verts, ari_vals):
    plt.scatter([xe] * len(ye), ye)
plt.title("kchange_qchange_ASE_KMeans")
plt.xlabel("n_verts")
plt.xticks(n_verts)
plt.ylabel("ARI")
plt.show()