# %% [markdown]
# #
from graspy.simulations import sbm
from graspy.plot import heatmap
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from graspy.utils import to_laplace
import matplotlib as mpl

sns.set_context("talk")
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False

# %% [markdown]
# #

n_per_comm = [100, 100, 100]
n_verts = np.sum(n_per_comm)
block_probs = np.array([[0.4, 0.1, 0.1], [0.1, 0.4, 0.1], [0.1, 0.1, 0.4]])

adj, labels = sbm(n_per_comm, block_probs, return_labels=True)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
sns.heatmap(
    block_probs, annot=True, cmap="RdBu_r", center=0, square=True, ax=axs[0], cbar=False
)
heatmap(adj, inner_hier_labels=labels, ax=axs[1], cbar=False)

#%%

I_DAD = to_laplace(adj, form="I-DAD")
DAD = to_laplace(adj, form="DAD")

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
heatmap_kws = dict(inner_hier_labels=labels, cbar=False)
heatmap(I_DAD, ax=axs[0], **heatmap_kws)
heatmap(DAD, ax=axs[1], **heatmap_kws)


def eig(A):
    evals, evecs = np.linalg.eig(A)
    sort_inds = np.argsort(evals)[::-1]
    evals = evals[sort_inds]
    evecs = evecs[:, sort_inds]
    return evals, evecs


fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)

evals, evecs = eig(I_DAD)
sns.scatterplot(np.arange(len(evals)), evals, ax=axs[0, 0])
axs[0, 0].set_ylabel("Eigenvalue")

evals, evecs = eig(DAD)
sns.scatterplot(np.arange(len(evals)), evals, ax=axs[0, 1])

U, S, V = np.linalg.svd(I_DAD)
sns.scatterplot(np.arange(len(S)), S, ax=axs[1, 0])
axs[1, 0].set_ylabel("Singular value")
axs[1, 0].set_xlabel("Index")

U, S, V = np.linalg.svd(DAD)
sns.scatterplot(np.arange(len(S)), S, ax=axs[1, 1])
axs[1, 1].set_xlabel("Index")

plt.tight_layout()


# %%
import pandas as pd

pal = sns.color_palette("Set1", n_colors=3)
evec_df = pd.DataFrame(data=U[:, :3])
evec_df["x"] = np.arange(n_verts)
evec_df["label"] = labels
fig, axs = plt.subplots(3, 2, figsize=(10, 6))
sns.scatterplot(data=evec_df, x="x", y=0, ax=axs[0, 1], hue="label", palette=pal)
sns.scatterplot(data=evec_df, x="x", y=1, ax=axs[1, 1], hue="label", palette=pal)
sns.scatterplot(data=evec_df, x="x", y=2, ax=axs[2, 1], hue="label", palette=pal)
