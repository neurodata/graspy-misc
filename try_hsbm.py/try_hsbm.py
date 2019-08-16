#%%
import numpy as np
from sklearn.metrics import adjusted_rand_score

from graspy.embed import AdjacencySpectralEmbed
from graspy.models import HSBMEstimator
from graspy.plot import heatmap, pairplot
from graspy.simulations import er_np, sbm

#%%


def n_to_labels(n):
    """Converts n vector (sbm input) to an array of labels
    
    Parameters
    ----------
    n : list or array
        length K vector indicating num vertices in each block
    
    Returns
    -------
    np.array
        shape (n_verts), indicator of what block each vertex 
        is in
    """
    n = np.array(n)
    n_cumsum = n.cumsum()
    labels = np.zeros(n.sum(), dtype=np.int64)
    for i in range(1, len(n)):
        labels[n_cumsum[i - 1] : n_cumsum[i]] = i
    return labels


B1 = np.array([[0.3, 0.25, 0.25], [0.25, 0.3, 0.25], [0.25, 0.25, 0.7]])
B2 = np.array([[0.4, 0.25, 0.25], [0.25, 0.4, 0.25], [0.25, 0.25, 0.4]])
B3 = np.array([[0.25, 0.2, 0.2], [0.2, 0.8, 0.2], [0.2, 0.2, 0.25]])

n = np.array([300, 600, 600, 600, 700, 600, 300, 400]).astype(float)
# n *= 1 / 5
n = n.astype(int)
block_labels = n_to_labels(n)
n_verts = np.sum(n)
global_p = 0.01
prop = np.array(
    [
        [0.4, 0.2, 0.4],
        [0.25, 0.5, 0.25],
        [0.25, 0.5, 0.25],
        [0.4, 0.2, 0.4],
        [0.25, 0.5, 0.25],
        [0.25, 0.5, 0.25],
        [0.25, 0.5, 0.25],
        [0.4, 0.2, 0.4],
    ]
)

B_list = [B1, B2, B3, B1, B3, B3, B2, B1]

graph = er_np(n_verts, global_p)
for i, n_sub_verts in enumerate(n):
    p = prop[i, :]
    n_vec = n_sub_verts * p
    n_vec = n_vec.astype(int)
    B = B_list[i]
    subgraph = sbm(n_vec, B)
    inds = block_labels == i
    graph[np.ix_(inds, inds)] = subgraph

heatmap(graph, figsize=(15, 15), cbar=False)

#%%
n_components = 8
ase = AdjacencySpectralEmbed(n_components=n_components)
latent = ase.fit_transform(graph)
pairplot(latent, labels=block_labels)

latent /= np.linalg.norm(latent, axis=1)[:, np.newaxis]
pairplot(latent, labels=block_labels)

from graspy.embed import ClassicalMDS

# def compute_cosine_similarity(latent):
#     for i in range(latent.shape[0])
similarity = latent @ latent.T
dissimilarity = 1 - similarity
print(dissimilarity[0, 0])
cmds = ClassicalMDS(n_components=n_components - 1, dissimilarity="precomputed")
cmds_latent = cmds.fit_transform(dissimilarity)
pairplot(cmds_latent, labels=block_labels)
#%%
hsbm = HSBMEstimator(n_subgraphs=8)
outer_labels, inner_labels = hsbm.fit(graph)


#%%
heatmap(
    graph,
    figsize=(15, 15),
    cbar=False,
    outer_hier_labels=inner_labels,
    inner_hier_labels=outer_labels,
)


adjusted_rand_score(block_labels, outer_labels)

