#%%
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import colorcet as cc

print(sns.__version__)

sns.set_context("talk")

palette_nums = {
    (0, 0): 0,
    (0, 1): 24,
    (0, 2): 38,
    (1, 0): 1,
    (1, 1): 214,
    (1, 2): 110,
    (2, 0): 157,
    (2, 1): 216,
    (2, 2): 186,
}

palette = {}
for key, val in palette_nums.items():
    palette[key] = cc.glasbey_light[val]


def plot_clustering(X, y, title=""):
    test_df = pd.DataFrame(data=X, columns=np.arange(X.shape[1], dtype=str))
    # test_df["labels"] = y_test.astype(str)
    test_df["labels"] = y

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    sns.scatterplot(
        data=test_df,
        x="0",
        y="1",
        hue="labels",
        ax=ax,
        palette=palette,
        hue_order=sorted(palette.keys()),
    )
    ax.get_legend().remove()
    ax.legend(
        bbox_to_anchor=(
            1,
            1,
        ),
        loc="upper left",
    )
    ax.set(title=title)
    ax.axis("off")
    return fig, ax


def predict(model, submodels, X):
    y1 = model.predict(X)
    y2 = np.empty(len(y1), dtype=int)
    # note that prediction of "0" in y corresponds to the mixture component with means_[0]
    # in the sklearn model
    for pred_label in range(model.n_components):
        # use the prediction at the first level to assign each point to a submodel for the
        # next level
        mask = y1 == pred_label
        X_sub = X[mask]
        submodel = submodels[pred_label]
        # use that submodel only for the prediction at that level
        y_sub = submodel.predict(X_sub)
        y2[mask] = y_sub
    y_total = list(zip(y1, y2))
    return y_total


#%% generate data
np.random.seed(888)
X, y = make_blobs(n_samples=400)
X_train, X_test, y_train, y_test = train_test_split(X, y)

#%% cluster
model = GaussianMixture(n_components=3)
model.fit(X_train)
y_pred = model.predict(X_train)

#%% subcluster
submodels = []
for pred_label in np.unique(y_pred):
    mask = y_pred == pred_label
    X_sub = X_train[mask]
    submodel = GaussianMixture(n_components=3)
    submodel.fit(X_sub)
    submodels.append(submodel)

#%% predict and sub predict on train
y_total = predict(model, submodels, X_train)
plot_clustering(X_train, y_total, title="Train data")

# %% predict and sub predict on test

y_total = predict(model, submodels, X_test)
plot_clustering(X_test, y_total, title="Test data")
