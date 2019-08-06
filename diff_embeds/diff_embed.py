from os.path import basename
from pathlib import Path
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

folderpath = Path(__file__.replace(basename(__file__), ""))
savepath = folderpath / "outputs" / "k_change_q_change" / "ASE" / "KMeans"

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

            #embedding and clustering
            Xhat = embed.fit_transform(G)
            clust = KMeans(n_clusters = k)
            labels_clust = clust.fit_predict(Xhat)
            ari = adjusted_rand_score(labels_sbm, labels_clust)
            temp.append(ari)
            print("n_verts: {} ARI: {}".format(n, ari))

        ari_vals.append(np.sum(temp) / n_sims)
        stand_error.append(stats.sem(temp))
        temp = []     
    return ari_vals, stand_error

#Variables
n_verts = [70, 105, 140, 175, 210, 245, 280, 315, 350]
slope = 35
p = 0.5
n_sims = 30
embed = LaplacianSpectralEmbed()

ari_vals, stand_error = avg_ari(slope, n_verts, n_sims, p, embed)
plt.errorbar(n_verts, 
             ari_vals, 
             yerr=stand_error,
             marker='s',
             mfc='red',
             mec='green')
plt.title(f"p = {p} q = {const_q}")
plt.xlabel("n_verts")
plt.xticks(n_verts)
plt.ylabel("ARI")
plt.show()
plt.savefig(folderpath / f"n{n_verts[0]}-{n_verts[len(n_verts)-1]}_sl{slope}_p{p}_s{n_sims}.pdf")