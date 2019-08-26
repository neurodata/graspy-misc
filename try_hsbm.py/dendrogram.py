import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster.hierarchy import dendrogram
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

    R = plot_dendrogram(model, no_plot=True, distance_sort="descending")
    inds = R["leaves"]
    distance_matrix = distance_matrix[np.ix_(inds, inds)]

    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        distance_matrix, cbar=False, square=True, xticklabels=False, yticklabels=False
    )

    divider = make_axes_locatable(ax)

    ax_x = divider.new_vertical(size=dendrogram_size, pack_start=False)
    ax.figure.add_axes(ax_x)
    plot_dendrogram(model, ax=ax_x, **dendrogram_kws)
    ax_x.axis("off")

    ax_y = divider.new_horizontal(size=dendrogram_size, pack_start=True)
    ax.figure.add_axes(ax_y)
    R = plot_dendrogram(model, ax=ax_y, orientation="left", **dendrogram_kws)
    ax_y.axis("off")
    ax_y.invert_yaxis()
    return ax
