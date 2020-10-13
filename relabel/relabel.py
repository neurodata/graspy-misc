#%%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

sns.set_context("talk")


X, y_true = make_blobs(n_samples=100, random_state=888888)
n_classes = len(np.unique(y_true))

palette = dict(zip(np.arange(n_classes), sns.color_palette("deep", n_classes)))

plot_df = pd.DataFrame(data=X, columns=np.arange(X.shape[1], dtype=str))
plot_df["true_labels"] = y_true


def simple_scatter(ax, hue, title=""):
    sns.scatterplot(data=plot_df, x="0", y="1", hue=hue, ax=ax, palette=palette)
    ax.set(xticks=[], yticks=[], ylabel="", xlabel="", title=title)
    ax.get_legend().remove()


fig, axs = plt.subplots(1, 3, figsize=(18, 6))
simple_scatter(axs[0], "true_labels", title="Known labeling")


gmm = GaussianMixture(n_components=3, random_state=80808)
y_predicted = gmm.fit_predict(X)
plot_df["predicted_labels"] = y_predicted

simple_scatter(axs[1], "predicted_labels", title="Predicted labeling (original)")


def remap_labels(y_true, y_pred, return_map=False):
    """
    Remaps a categorical labeling (such as one predicted by a clustering algorithm) to
    match the labels used by another similar labeling.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels, or, labels to map to.
    y_pred : array-like of shape (n_samples,)
        Labels to remap to match the categorical labeling of `y_true`.

    Returns
    -------
    remapped_y_pred : np.ndarray of shape (n_samples,)
        Same categorical labeling as that of `y_pred`, but with the category labels
        permuted to best match those of `y_true`.
    label_map : dict
        Mapping from the original labels of `y_pred` to the new labels which best
        resemble those of `y_true`.

    Examples
    --------
    >>> y_true = np.array([0,0,1,1,2,2])
    >>> y_pred = np.array([2,2,1,1,0,0])
    >>> remap_labels(y_true, y_pred)
    array([0, 0, 1, 1, 2, 2])

    """
    confusion_mat = confusion_matrix(y_true, y_pred)
    row_inds, col_inds = linear_sum_assignment(confusion_mat, maximize=True)
    label_map = dict(zip(col_inds, row_inds))
    remapped_y_pred = np.vectorize(label_map.get)(y_pred)
    if return_map:
        return remapped_y_pred, label_map
    else:
        return remapped_y_pred


y_remapped = remap_labels(y_true, y_predicted)
plot_df["remapped_labels"] = y_remapped
simple_scatter(axs[2], "remapped_labels", title="Predicted labeling (remapped)")

axs[2].legend(
    bbox_to_anchor=(1, 1),
    loc="upper left",
)
