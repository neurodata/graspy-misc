import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
from scipy import stats
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

#Generate B Matrix
def B_matrix(k, p, q):
    B = np.zeros((k,k))
    np.fill_diagonal(B, p)
    B[B == 0] = q
    return B

def avg_ari(slope, n_verts, n_sims, p, embed):
    temp = []
    ari_vals = []
    stand_error = []
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
            print("{}: {}".format(n, ari))
        ari_vals.append(np.sum(temp) / n_sims)
        stand_error.append(stats.sem(temp))
        temp = []     
    return(ari_vals, stand_error)

#Variables
slope = 25
p = 0.5
n_verts = [50, 75, 100, 125, 150, 175, 200]
n_sims = 35
embed = AdjacencySpectralEmbed()

ari_vals, stand_error = avg_ari(slope, n_verts, n_sims, p, embed)
print(ari_vals)
print(stand_error)
plt.errorbar(n_verts, 
             ari_vals, 
             yerr=stand_error,
             marker='s',
             mfc='red',
             mec='green')
plt.title("kchange_qchange_ASE_KMeans")
plt.xlabel("n_verts")
plt.xticks(n_verts)
plt.ylabel("ARI")
plt.show()