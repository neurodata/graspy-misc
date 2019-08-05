import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns 
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
slope = 100
p = 0.75
n_verts = [200, 300, 400]
n_sims = 1
embed = AdjacencySpectralEmbed()

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
        heatmap(G, title=f"k={k} q={q}", inner_hier_labels=labels_sbm)

        #embedding
        Xhat = embed.fit_transform(G)
        pairplot(Xhat, title="Adjacency Spectral Embedding")

        #clustering
        clust = KMeans(n_clusters = k)
        palette = {'Right':(0,0.7,0.2),
                   'Wrong':(0.8,0.1,0.1)}
        labels_clust = clust.fit_predict(Xhat)
        ari = adjusted_rand_score(labels_sbm, labels_clust)
        print(labels_sbm)
        print(labels_clust)
        error = labels_sbm - labels_clust
        error = error != 0
        if np.sum(error) / (n) > 0.5:
                error = error == 0
        error_rate = np.sum(error) / (n)
        error_label = (n) * ["Right"]
        error_label = np.array(error_label)
        error_label[error]= "Wrong"

        pairplot(Xhat,
                 labels=labels_clust,
                 title='KMeans on embedding, ARI: {}'.format(str(ari)[:5]),
                 legend_name='Predicted label',
                 height=3.5,
                 palette='muted')
        
        pairplot(Xhat,
                 labels=error_label,
                 title='Error from KMeans, Error rate: {}'.format(str(error_rate)),
                 legend_name='Error label',
                 height=3.5,
                 palette=palette)
        plt.show()