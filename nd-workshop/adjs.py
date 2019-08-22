#%%
import numpy as np
import matplotlib.pyplot as plt
from graspy.plot import heatmap
from graspy.datasets import load_drosophila_right
from graspy.utils import binarize

graph, labels = load_drosophila_right(return_labels=True)
graph = binarize(graph)
inds = np.random.permutation(graph.shape[0])

perm_graph = graph[np.ix_(inds, inds)]
plt.style.use("seaborn-white")
fig, ax = plt.subplots(1, 3, figsize=(30, 12))
heatmap(perm_graph, cbar=False, ax=ax[0])
heatmap(perm_graph, sort_nodes=True, cbar=False, ax=ax[1])
heatmap(graph, inner_hier_labels=labels, sort_nodes=True, cbar=False, ax=ax[2])
plt.tight_layout()
plt.savefig("./multi-adj-dros.png", facecolor="w")
#%%
