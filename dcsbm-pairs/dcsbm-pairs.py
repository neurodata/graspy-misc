#%%

from graspy.simulations import sbm
import numpy as np
from graspy.plot import heatmap, pairplot

n = np.array([100, 100, 100])
p = np.array([[0.3, 0.2, 0.1], [0.01, 0.2, 0.2], [0.02, 0.03, 0.1]])
dcs = []
for i in range(len(n)):
    dc = np.random.beta(2, 5, n[i])
    dc /= dc.sum()
    dcs.append(dc)
dcs = np.concatenate(dcs)
adj, labels = sbm(n, p, directed=True, dc=dcs, return_labels=True)
heatmap(adj, cbar=False, sort_nodes=True, inner_hier_labels=labels)

#%%
from graspy.embed import AdjacencySpectralEmbed

ase = AdjacencySpectralEmbed(n_components=3)
embed = ase.fit_transform(adj)
embed = np.concatenate(embed, axis=-1)

#%%
pairplot(embed, labels=labels)

# %% [markdown]
# ##

norm_embed = embed / np.linalg.norm(embed, axis=1)[:, None]
pairplot(norm_embed, labels=labels)

# %% [markdown]
# ##
import matplotlib.pyplot as plt

n_dim = norm_embed.shape[1]
fig, axs = plt.subplots(n_dim, n_dim, figsize=(10, 10))
for i in range(n_dim):
    for j in range(n_dim):
        if i != j:
            ax = axs[i, j]
            ax.axis("off")
            sns.scatterplot()

# %% [markdown]
# ##
import n_sphere

norm_embed_spherical = n_sphere.convert_spherical(norm_embed)
norm_embed_spherical = norm_embed_spherical[:, 1:]  # chop off R dimension
pairplot(norm_embed_spherical, labels=labels)
