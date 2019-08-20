from os.path import basename
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np 
from scipy import stats
from graspy.simulations import sbm
from graspy.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed
from graspy.plot import heatmap
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from src.utils import n_to_labels

folderpath = Path(__file__.replace(basename(__file__), ""))
savepath = folderpath / "outputs" 

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
            k = const_k
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
n_verts = [80, 120, 160, 200, 240, 280, 320, 360, 400]
slope = n_verts[0] / 2
print(slope)
p = 0.5
n_sims = 30
embed = AdjacencySpectralEmbed()
embed1 = LaplacianSpectralEmbed()
ari_vals, stand_error = avg_ari(slope, n_verts, n_sims, p, embed)
ari_vals1, stand_error1 = avg_ari(slope, n_verts, n_sims, p, embed1)

plt.errorbar(n_verts, 
             ari_vals, 
             yerr=stand_error,
             marker='s',
             mfc='red',
             label="ASE")
plt.errorbar(n_verts, 
             ari_vals1, 
             yerr=stand_error1,
             marker='s',
             mfc='blue',
             label="LSE")
plt.title(f"k = n_verts / {slope}, p = {p}, q = {slope} / n_verts")
plt.xlabel("n_verts")
plt.xticks(n_verts)
plt.ylabel("ARI")
plt.legend(loc='upper left')
plt.savefig(savepath / "kconst_qchange_ASEvsLSE_KMeans.pdf")
plt.show()