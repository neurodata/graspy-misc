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

def avg_ari(slope, n_verts, n_sims, p, embed, const_k=True, const_q=True):
    temp = []
    ari_vals = []
    stand_error = []
        
    #Generate graph
    for n in n_verts:
        for _ in range(n_sims):
            
            if const_k:
                k, k_func = constant_k
            else:
                k = linear_k(slope, n)
                k_func = f"n_verts / {slope}"
                
            if const_q:
                q, q_func = constant_q
            else:
                q = decay_q(slope, n)
                q_func = f"{slope} / n_verts"
                
            B = B_matrix(k, p, q)
            cn = [n//k] * k
            labels_sbm = n_to_labels(cn).astype(int)
            G = sbm(cn, B)

            #embedding and clustering
            Xhat = embed.fit_transform(G)
            
            #can be KMeans or GMM
            clust = KMeans(n_clusters = k)
            labels_clust = clust.fit_predict(Xhat)
            ari = adjusted_rand_score(labels_sbm, labels_clust)
            temp.append(ari)
            print(f"n_verts: {n} ARI: {ari}")

        ari_vals.append(np.sum(temp) / n_sims)
        stand_error.append(stats.sem(temp))
        temp = []     
    return ari_vals, stand_error, k_func, q_func

n_verts = [80, 120, 160, 200, 240, 280, 320, 360, 400]
slope = n_verts[0] / 2
p = 0.5
n_sims = 30
embed_ase = AdjacencySpectralEmbed()
embed_lse = LaplacianSpectralEmbed()

ari_vals_ase, stand_error_ase, k_func, q_func = avg_ari(slope, n_verts, n_sims, p, embed_ase, const_k=False, const_q=False)
ari_vals_lse, stand_error_lse, _, _ = avg_ari(slope, n_verts, n_sims, p, embed_lse, const_k=False, const_q=False)

plt.errorbar(n_verts, 
             ari_vals_ase, 
             yerr=stand_error_ase,
             marker='s',
             mfc='red',
             label="ASE")
plt.errorbar(n_verts, 
             ari_vals_lse, 
             yerr=stand_error_lse,
             marker='s',
             mfc='blue',
             label="LSE")
plt.title(f"k = {k_func}, p = {p}, q = {q_func}") 
plt.xlabel("n_verts")
plt.xticks(n_verts)
plt.ylabel("ARI")
plt.legend(loc='upper left')
plt.savefig(savepath / "kconst_qconst_ASEvsLSE_KMeans.pdf")