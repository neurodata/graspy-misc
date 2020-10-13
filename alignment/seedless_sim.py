#%%
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hyppo.discrim import DiscrimOneSample
from scipy.stats import ortho_group
from sklearn.metrics import pairwise_distances

from graspy.align import SeedlessProcrustes
from graspy.inference import LatentDistributionTest
from graspy.simulations import p_from_latent, sample_edges
from graspy.utils import symmetrize

np.random.seed(8888)

sns.set_context("talk")

mpl.rcParams["axes.edgecolor"] = "lightgrey"
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False


def hardy_weinberg(theta):
    """
    Maps a value from [0, 1] to the hardy weinberg curve.
    """
    hw = [theta ** 2, 2 * theta * (1 - theta), (1 - theta) ** 2]
    return np.array(hw).T


def sample_hw_graph(thetas):
    latent = hardy_weinberg(thetas)
    p_mat = p_from_latent(latent, rescale=False, loops=False)
    graph = sample_edges(p_mat, directed=False, loops=False)
    return (graph, p_mat, latent)


def scatter(X, ax, s=10, linewidth=0, alpha=0.5, title="", **kws):
    plot_df = pd.DataFrame(X)
    sns.scatterplot(
        data=plot_df, x=0, y=1, s=s, linewidth=linewidth, alpha=alpha, ax=ax, **kws
    )
    soft_axes_off(ax)
    ax.set(title=title)


def soft_axes_off(ax):
    ax.set(xticks=[], yticks=[], ylabel="", xlabel="")


# %% generate data

n_samples = 100
thetas = np.random.uniform(0.2, 0.8, n_samples)
manifold_points = hardy_weinberg(thetas)
sigma = 0.00005
n_dimensions = 3

resample_manifold_points = True

X = manifold_points + np.random.multivariate_normal(
    [0, 0, 0], sigma * np.eye(n_dimensions), n_samples
)

if resample_manifold_points:
    thetas = np.random.uniform(0.2, 0.8, n_samples)
    manifold_points = hardy_weinberg(thetas)

Y = manifold_points + np.random.multivariate_normal(
    [0, 0, 0], sigma * np.eye(n_dimensions), n_samples
)

Q = ortho_group.rvs(n_dimensions)
X_rot = X @ Q

#%% example run
random_Q = ortho_group.rvs(n_dimensions)
sp = SeedlessProcrustes(initialization="custom", initial_Q=random_Q)
sp.fit(X_rot, Y)
Q_hat_inv = sp.Q_X

colors = sns.color_palette("deep", 10)

fig, axs = plt.subplots(2, 3, figsize=(12, 8))
axs = axs.ravel()

ax = axs[0]
scatter(Y, ax, color=colors[0], title=r"$Y$")

ax = axs[1]
scatter(X, ax, color=colors[1], title=r"$X$")

ax = axs[2]
scatter(X_rot, ax, color=colors[1], title=r"$XQ$")

ax = axs[3]
ax.axis("off")

ax = axs[4]
scatter(X_rot @ Q.T, ax, color=colors[1], title=r"$X Q^T$")

ax = axs[5]
scatter(X_rot @ Q_hat_inv, ax, color=colors[1], title=r"$X \hat{Q}^T$")

error_norm = np.linalg.norm(X_rot @ Q.T - X_rot @ Q_hat_inv)
print(error_norm)

# %% [markdown]
# ##
n_init = 10
for i in range(n_init):
    random_Q = ortho_group.rvs(n_dimensions)
    sp = SeedlessProcrustes(initialization="custom", initial_Q=random_Q)
    sp.fit(X_rot, Y)
    Q_hat_inv = sp.Q_X
    error_norm = np.linalg.norm(X_rot @ Q.T - X_rot @ Q_hat_inv)
    print(error_norm)

# %% [markdown]
# ##

# %% [markdown]
# ## TODO
# - write down the generative model
# - run multiple orthogonal initializations for a single set of points, take the best
# - have one plot showing error norm for each of the 3 initialization methods
# - in a model where points have a true correspondence (e.g. 2 noisy observations from
# same point on the manifold, or two vertices that are corr-RDPG) look at "recall by k"
