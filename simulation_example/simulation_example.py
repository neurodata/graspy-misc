# %% [markdown]
# #
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
import pandas as pd
import matplotlib as mpl

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
sns.set_context("talk")

cov1 = np.array([[1, 0], [0, 1]])
mu1 = np.array([1, 1])
cov2 = cov1
mu2 = -mu1
pi = [0.5, 0.5]
n_samples = 200
X1 = np.random.multivariate_normal(mu1, cov1, int(pi[0] * n_samples))
X2 = np.random.multivariate_normal(mu2, cov2, int(pi[1] * n_samples))
X = np.concatenate((X1, X2))
y = np.array(len(X1) * ["1"] + len(X2) * ["2"])

plot_df = pd.DataFrame(data=X, columns=["x", "y"])
plot_df["Label"] = y

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
sns.scatterplot(data=plot_df, hue="Label", x="x", y="y", ax=ax)

# %% [markdown]
# #
kmeans = KMeans(n_clusters=2)
gmm = GaussianMixture(n_components=2, covariance_type="full")

kmeans_labels = kmeans.fit_predict(X)
gmm_labels = gmm.fit_predict(X)
kmeans_ari = adjusted_rand_score(y, kmeans_labels)
gmm_ari = adjusted_rand_score(y, gmm_labels)

# %% [markdown]
# #


def fit_and_score(estimator, X, y, name):
    pred_labels = estimator.fit_predict(X)
    ari = adjusted_rand_score(y, pred_labels)
    result = {"ARI": ari, "Method": name}
    return result


props = np.linspace(0.5, 0.9, 9)
rows = []
n_sims = 20
n_samples = 300

for sim in range(n_sims):
    for prop in props:
        pi = [prop, 1 - prop]
        X1 = np.random.multivariate_normal(mu1, cov1, int(pi[0] * n_samples))
        X2 = np.random.multivariate_normal(mu2, cov2, int(pi[1] * n_samples))
        X = np.concatenate((X1, X2))
        y = np.array(len(X1) * ["1"] + len(X2) * ["2"])
        result = fit_and_score(kmeans, X, y, "KMeans")
        result["Proportion"] = prop
        rows.append(result)
        result = fit_and_score(gmm, X, y, "GMM")
        result["Proportion"] = prop
        rows.append(result)

result_df = pd.DataFrame(rows)

fig, axs = plt.subplots(2, 1, figsize=(8, 8))
sns.stripplot(data=result_df, x="Proportion", y="ARI", hue="Method", ax=axs[0])
axs[0].set_xticks([])
axs[0].set_xlabel("")
sns.lineplot(data=result_df, x="Proportion", y="ARI", hue="Method", ax=axs[1])
fig.suptitle("2 spherical covariance Gaussians")
