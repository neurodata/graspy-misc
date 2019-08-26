#%%
import numpy as np
from sklearn.metrics import adjusted_rand_score

from graspy.embed import AdjacencySpectralEmbed
from graspy.models import HSBMEstimator
from graspy.plot import heatmap, pairplot
from graspy.simulations import er_np, sbm

#%%
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris
from sklearn.metrics.pairwise import euclidean_distances


def plot_dendrogram(model, **kwargs):

    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0] + 2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(
        float
    )

    # Plot the corresponding dendrogram
    return dendrogram(linkage_matrix, **kwargs)


def plot_agglomerative(
    model, distance_matrix, dendrogram_size="45%", figsize=(10, 10), dendrogram_kws={}
):
    dendrogram_kws["distance_sort"] = "descending"
    dendrogram_kws["color_threshold"] = 0
    dendrogram_kws["above_threshold_color"] = "k"

    inds = np.triu_indices_from(distance_matrix, k=1)
    condensed_distances = distance_matrix[inds]
    linkage_mat = linkage(condensed_distances, method="average")
    R = dendrogram(linkage_mat, no_plot=True, distance_sort="descending")
    # R = plot_dendrogram(model, no_plot=True, distance_sort="descending")
    inds = R["leaves"]
    distance_matrix = distance_matrix[np.ix_(inds, inds)]

    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        distance_matrix, cbar=False, square=True, xticklabels=False, yticklabels=False
    )

    divider = make_axes_locatable(ax)

    ax_x = divider.new_vertical(size=dendrogram_size, pack_start=False, pad=0.1)
    ax.figure.add_axes(ax_x)
    # plot_dendrogram(model, ax=ax_x, **dendrogram_kws)
    dendrogram(linkage_mat, ax=ax_x, **dendrogram_kws)
    ax_x.axis("off")

    ax_y = divider.new_horizontal(size=dendrogram_size, pack_start=True, pad=0.1)
    ax.figure.add_axes(ax_y)
    # R = plot_dendrogram(model, ax=ax_y, orientation="left", **dendrogram_kws)
    dendrogram(linkage_mat, ax=ax_y, orientation="left", **dendrogram_kws)
    ax_y.axis("off")
    ax_y.invert_yaxis()
    return ax


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
hsbm = HSBMEstimator(n_subgraphs=8, n_subgroups=3)
hsbm.fit(graph)
#%%
plt.style.use("seaborn-white")
model = hsbm.agglomerative_model_
dists = hsbm.subgraph_dissimilarities_
dists = dists - dists.min()
plot_agglomerative(model, dists, dendrogram_size="30%")
#%%
data = load_iris().data
dists = euclidean_distances(data)
plot_agglomerative(model, dists, figsize=(20, 20))
#%%
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage

plt.figure(figsize=(10, 10))
cond_inds = np.triu_indices_from(dists, k=1)
cond_dists = dists[cond_inds]
Z = linkage(cond_dists, method="average")
# Z[:, 2] = 1
dendrogram(Z)
#%% plot the whole graph sorted by found communities and motifs
inner_labels = hsbm.vertex_assignments_
outer_labels = hsbm.subgraph_types_
heatmap(
    graph,
    figsize=(15, 15),
    cbar=False,
    outer_hier_labels=inner_labels,
    inner_hier_labels=outer_labels,
)


adjusted_rand_score(block_labels, outer_labels)


#%%
