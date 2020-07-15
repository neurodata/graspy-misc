#%%
import os
from timeit import timeit

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import rand
from scipy.sparse.construct import random
from scipy.sparse.linalg import svds
from sklearn.utils.extmath import cartesian, randomized_svd
from tqdm import tqdm

from src.io import savefig

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


densities = np.geomspace(0.0001, 0.01, 3)
print(densities)
# sizes = np.geomspace(100, 10000, 5)
sizes = [512, 1024, 2048, 4096]
sizes = np.array(sizes, dtype=float)
print(sizes)
ranks = [4, 8, 16, 32]
n_init = 10
rows = []


# %%
for (n, density, rank) in tqdm(cartesian((sizes, densities, ranks,))):
    n = int(n)
    rank = int(rank)
    for init in range(n_init):
        matrix = rand(n, n, density=density, format="csr")
        time = timeit("randomized_svd(matrix, rank)", globals=globals(), number=1)
        rows.append(
            dict(
                n=n,
                density=density,
                rank=rank,
                time=time,
                method="randomized",
                init=init,
            )
        )
        time = timeit("svds(matrix, rank)", globals=globals(), number=1)
        rows.append(
            dict(
                n=n, density=density, rank=rank, time=time, method="sparse", init=init,
            )
        )
results = pd.DataFrame(rows)
results.to_csv(
    "GraphEmbeddingMethods/sandbox/outs/2020-07-07-svd-profile/time-results.csv"
)
# %%

rank = ranks[0]
density = densities[0]
plot_results = results[(results["rank"] == rank) & (results["density"] == density)]

jitter_results = plot_results.copy()
# TODO log scale the jitter
jitter_results["n"] = jitter_results["n"] + np.random.normal(0, 10, len(jitter_results))
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.scatterplot(
    data=jitter_results, x="n", y="time", hue="method", alpha=0.5, linewidth=0, s=10
)
sns.lineplot(data=plot_results, x="n", y="time", hue="method")
ax.set_xscale("log")
stashfig(f"profile-rank={rank}-density={density}")

# %%


sns.set_context("talk")
mpl.rcParams["xtick.major.size"] = 0
mpl.rcParams["ytick.major.size"] = 0
mpl.rcParams["axes.edgecolor"] = "lightgrey"
sparse_results = results[results["method"] == "sparse"].reset_index()
randomized_results = results[results["method"] == "randomized"].reset_index()
diff_results = randomized_results.copy().drop("time", axis=1)
diff_results["time_diff"] = randomized_results["time"] / sparse_results["time"]
#     sparse_results["time"] - randomized_results["time"]
# ) / sparse_results["time"]
fg = sns.FacetGrid(
    data=diff_results,
    row="density",
    col="rank",
    sharey=True,
    despine=True,
    margin_titles=True,
    ylim=(0, 2),
)
fg.map(sns.lineplot, "n", "time_diff")
fg.map(sns.scatterplot, "n", "time_diff", alpha=0.5, linewidth=0, s=10)


def add_lines(*args, **kwargs):
    ax = plt.gca()
    ax.axhline(1, linewidth=1, linestyle="--", color="darkred")
    ax.set_yticks([0, 1, 2])


fg.fig.text(
    -0.02, 0.5, "rsvd / svds time", rotation=90, fontsize=24, va="center", ha="center",
)
fg.fig.text(0.5, 0, "N", fontsize=24, va="center", ha="center")
ax = fg.axes[0, 0]
# ax.text(10000, 0.3, "rsvd faster", ha="right")
# ax.text(10000, 1.7, "svds faster", ha="right")
fg.map(add_lines)
stashfig("svd-compare-time")

# %%
from scipy.sparse.linalg import norm

for (n, density, rank) in tqdm(cartesian((sizes, densities, ranks,))):
    n = int(n)
    rank = int(rank)
    for init in range(n_init):
        matrix = rand(n, n, density=density, format="csr")
        U, S, Vt = randomized_svd(matrix, rank,)
        randomized_error = np.linalg.norm(
            U @ np.diag(S) @ Vt - matrix.todense(), ord="fro"
        )
        rows.append(
            dict(
                n=n,
                density=density,
                rank=rank,
                error=randomized_error,
                method="randomized",
                init=init,
            )
        )
        U, S, Vt = svds(matrix, rank)
        sparse_error = np.linalg.norm(U @ np.diag(S) @ Vt - matrix.todense(), ord="fro")
        if np.isnan(sparse_error):
            print("nan")
        rows.append(
            dict(
                n=n,
                density=density,
                rank=rank,
                error=sparse_error,
                method="sparse",
                init=init,
            )
        )
results = pd.DataFrame(rows)
results.to_csv(
    "GraphEmbeddingMethods/sandbox/outs/2020-07-07-svd-profile/errors-results.csv"
)

# %%

sparse_results = results[results["method"] == "sparse"].reset_index()
randomized_results = results[results["method"] == "randomized"].reset_index()
diff_results = randomized_results.copy().drop("error", axis=1)
diff_results["error_ratio"] = randomized_results["error"] / sparse_results["error"]
fg = sns.FacetGrid(
    data=diff_results,
    row="density",
    col="rank",
    sharey=False,
    despine=True,
    margin_titles=True,
    # ylim=(0.999, 1.01),
)
fg.map(sns.lineplot, "n", "error_ratio")
fg.map(sns.scatterplot, "n", "error_ratio", alpha=0.5, linewidth=0, s=10)


def add_lines(*args, **kwargs):
    ax = plt.gca()
    ax.axhline(1, linewidth=1, linestyle="--", color="darkred")
    # ax.set_yticks([0, 1, 2])


fg.fig.text(
    -0.02, 0.5, "rsvd / svds error", rotation=90, fontsize=24, va="center", ha="center",
)
fg.fig.text(0.5, 0, "N", fontsize=24, va="center", ha="center")
ax = fg.axes[0, 0]
# ax.text(10000, 0.3, "rsvd better", ha="right")
# ax.text(10000, 1.7, "svds better", ha="right")
fg.map(add_lines)
stashfig("svd-compare-error")

