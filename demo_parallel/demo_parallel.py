# %% [markdown]
# # Imports

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

# %% [markdown]
# # Simple for loop to get random numbers

np.random.seed(8888)

n_numbers = 20

outs = []
for i in range(n_numbers):
    big_number = np.random.randint(1e8)
    outs.append(big_number)
print(outs)

# %% [markdown]
# # Make the above stuff in my for loop into a function


def _get_big_number():
    big_number = np.random.randint(1e8)
    return big_number


np.random.seed(8888)

outs = []
for i in range(n_numbers):
    big_number = _get_big_number()
    outs.append(big_number)

print(outs)


# %% [markdown]
# # Do the same thing but in parallel

np.random.seed(8888)
par = Parallel(n_jobs=8)
outs = par(delayed(_get_big_number)() for _ in range(n_numbers))
print(outs)  # note that now we don't get reproducible results!


# %% [markdown]
# # Get random numbers in parallel, reproducibly

np.random.seed(8888)
seeds = np.random.randint(1e8, size=n_numbers)


def _get_big_reproducible_number(seed):
    np.random.seed(seed)
    return _get_big_number()


par = Parallel(n_jobs=4)
outs = par(delayed(_get_big_reproducible_number)(seed) for seed in seeds)
print(outs)

# %% [markdown]
# # Simple demo with Gaussian blobs


def generate_data(n_samples=300):
    X, y = make_blobs(n_samples=n_samples, cluster_std=2)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)
    return aniso


X, y = generate_data()
plot_df = pd.DataFrame(data=X)
plot_df["Label"] = y

sns.set_context("talk")
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
sns.scatterplot(
    data=plot_df,
    x=0,
    y=1,
    ax=ax,
    hue="Label",
    palette=sns.color_palette("Set1", plot_df["Label"].nunique()),
)
ax.axis("off")

# %% [markdown]
# # Look at the performance of two different clustering algorithms

gmm = GaussianMixture(n_components=3, covariance_type="full")
gmm_pred_labels = gmm.fit_predict(X)

gmm_ari = adjusted_rand_score(y, gmm_pred_labels)
print(f"GMM ARI:{gmm_ari}")

kmeans = KMeans(n_clusters=3)
kmeans_pred_labels = kmeans.fit_predict(X)
kmeans_ari = adjusted_rand_score(y, kmeans_pred_labels)
print(f"K-means ARI: {kmeans_ari}")

plot_df["KMeans"] = kmeans_pred_labels
plot_df["GMM"] = gmm_pred_labels

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
sns.scatterplot(
    data=plot_df,
    x=0,
    y=1,
    hue="KMeans",
    ax=axs[0],
    palette=sns.color_palette("Set1", plot_df["KMeans"].nunique()),
)
sns.scatterplot(
    data=plot_df,
    x=0,
    y=1,
    hue="GMM",
    ax=axs[1],
    palette=sns.color_palette("Set1", plot_df["KMeans"].nunique()),
)
axs[0].axis("off")
axs[0].set_title(f"ARI: {kmeans_ari}")
axs[1].axis("off")
axs[1].set_title(f"ARI: {gmm_ari}")


# %% [markdown]
# # Now run an actual experiment over many random inits


def run_experiment(seed):
    np.random.seed(seed)
    X, y = generate_data()

    gmm = GaussianMixture(n_components=3, covariance_type="full", n_init=10)
    gmm_pred_labels = gmm.fit_predict(X)
    gmm_ari = adjusted_rand_score(y, gmm_pred_labels)

    kmeans = KMeans(n_clusters=3)
    kmeans_pred_labels = kmeans.fit_predict(X)
    kmeans_ari = adjusted_rand_score(y, kmeans_pred_labels)

    return {"KMeans": kmeans_ari, "GMM": gmm_ari}


np.random.seed(8888)
n_sims = 10
seeds = np.random.randint(1e8, size=n_sims)  # random
# seeds = np.ones(n_sims, dtype=int) # not random
par = Parallel(n_jobs=2)
outs = par(delayed(run_experiment)(seed) for seed in seeds)
ari_df = pd.DataFrame(outs)
ari_df = ari_df.melt(var_name="Method", value_name="ARI")

sns.stripplot(data=ari_df, x="Method", y="ARI")

