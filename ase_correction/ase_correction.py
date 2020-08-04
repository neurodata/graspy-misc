#%%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.defchararray import replace
import pandas as pd
import seaborn as sns
from graspy.embed import AdjacencySpectralEmbed
from graspy.plot import heatmap
from graspy.simulations import sbm
from numpy.core.shape_base import block
from sklearn.mixture import GaussianMixture

sns.set_context("talk")


n_per_comm = [1000, 1000, 1000]
n_verts = np.sum(n_per_comm)
block_probs = np.array([[0.5, 0.1, 0.1], [0.1, 0.5, 0.1], [0.1, 0.1, 0.5]])


adj, labels = sbm(n_per_comm, block_probs, return_labels=True)


# %%

ase = AdjacencySpectralEmbed(n_components=3)
Xhat = ase.fit_transform(adj)


# %%


# REF: Anton
def _fit_plug_in_variance_estimator(X):
    """
    Takes in ASE of a graph and returns a function that estimates
    the variance-covariance matrix at a given point using the
    plug-in estimator from the RDPG Central Limit Theorem.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        adjacency spectral embedding of a graph

    Returns
    -------
    plug_in_variance_estimtor: functions
        a function that estimates variance (see below)
    """

    n = len(X)

    # precompute the Delta and the middle term matrix part
    delta = 1 / (n) * (X.T @ X)
    delta_inverse = np.linalg.inv(delta)
    middle_term_matrix = np.einsum("bi,bo->bio", X, X)

    def plug_in_variance_estimator(x):
        """
        Takes in a point of a matrix of points in R^d and returns an
        estimated covariance matrix for each of the points

        Parameters:
        -----------
        x: np.ndarray, shape (n, d)
            points to estimate variance at
            if 1-dimensional - reshaped to (1, d)

        Returns:
        -------
        covariances: np.ndarray, shape (n, d, d)
            n estimated variance-covariance matrices of the points provided
        """
        if x.ndim < 2:
            x = x.reshape(1, -1)
        # the following two lines are a properly vectorized version of
        # middle_term = 0
        # for i in range(n):
        #     middle_term += np.multiply.outer((x @ X[i] - (x @ X[i]) ** 2),
        #                                      np.outer(X[i], X[i]))
        # where the matrix part does not involve x and has been computed above
        middle_term_scalar = x @ X.T - (x @ X.T) ** 2
        # print(middle_term_scalar)
        middle_term = np.tensordot(middle_term_scalar, middle_term_matrix, axes=1)
        covariances = delta_inverse @ (middle_term / n) @ delta_inverse
        return covariances

    return plug_in_variance_estimator


# plug_in_estimator = _fit_plug_in_variance_estimator(Xhat)

# covs = plug_in_estimator(Xhat)


def make_ellipse(vec, cov, ax, alpha=0.5, **kws):
    v, w = np.linalg.eigh(cov)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan2(u[1], u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    ell = mpl.patches.Ellipse(vec, v[0], v[1], 180 + angle, **kws)
    ell.set_clip_box(ax.bbox)
    ell.set_alpha(alpha)
    ax.add_artist(ell)


plot_embed = pd.DataFrame(data=Xhat)
plot_embed["label"] = labels.astype("str")
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
sns.scatterplot(
    data=plot_embed,
    x=0,
    y=1,
    hue="label",
    palette=sns.color_palette("deep", plot_embed["label"].nunique()),
    s=5,
    linewidth=0,
    alpha=0.5,
)

plug_in_estimator = _fit_plug_in_variance_estimator(Xhat)
covs = plug_in_estimator(Xhat) * (N - M) / (N * M)

n_ellipses = 15
choice_inds = np.random.choice(len(Xhat), n_ellipses, replace=False)
for i, (x, cov) in enumerate(zip(Xhat, covs)):
    if i in choice_inds:
        make_ellipse(x, cov, ax=ax, alpha=0.4, fill=False)

ax.axis("off")
ax.get_legend().remove()
# ax.set(xlim=(-2, 3), ylim=(-2, 3))
ax.legend(bbox_to_anchor=(1, 1,), loc="upper left")

# %% see what the ellipses look like for GMM

gmm = GaussianMixture(n_components=3)
gmm.fit(Xhat)


plot_embed = pd.DataFrame(data=Xhat)
plot_embed["label"] = labels.astype("str")
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
sns.scatterplot(
    data=plot_embed,
    x=0,
    y=1,
    hue="label",
    palette=sns.color_palette("deep", plot_embed["label"].nunique()),
    s=5,
    linewidth=0,
    alpha=0.5,
)


means = gmm.means_
covs = gmm.covariances_

for x, cov in zip(means, covs):
    make_ellipse(x, cov, ax=ax, alpha=1, fill=False)

ax.axis("off")
ax.get_legend().remove()
ax.legend(bbox_to_anchor=(1, 1,), loc="upper left")


# %%


import warnings

import numpy as np
from scipy import stats

from graspy.embed import select_dimension, AdjacencySpectralEmbed
from graspy.utils import import_graph
from graspy.inference.base import BaseInference
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics.pairwise import PAIRED_DISTANCES
from sklearn.metrics.pairwise import PAIRWISE_KERNEL_FUNCTIONS
from hyppo.ksample import KSample
from hyppo._utils import gaussian

_VALID_DISTANCES = list(PAIRED_DISTANCES.keys())
_VALID_KERNELS = list(PAIRWISE_KERNEL_FUNCTIONS.keys())
_VALID_KERNELS.append("gaussian")  # can use hyppo's medial gaussian kernel too
_VALID_METRICS = _VALID_DISTANCES + _VALID_KERNELS

_VALID_TESTS = ["cca", "dcorr", "hhg", "rv", "hsic", "mgc"]


class LatentDistributionTest(BaseInference):
    """
    Two-sample hypothesis test for the problem of determining whether two random
    dot product graphs have the same distributions of latent positions.

    This test can operate on two graphs where there is no known matching
    between the vertices of the two graphs, or even when the number of vertices
    is different. Currently, testing is only supported for undirected graphs.

    Read more in the :ref:`tutorials <inference_tutorials>`

    Parameters
    ----------
    test : str
        Backend hypothesis test to use, one of ["cca", "dcorr", "hhg", "rv", "hsic", "mgc"].
        These tests are typically used for independence testing, but here they
        are used for a two-sample hypothesis test on the latent positions of
        two graphs. See :class:`hyppo.ksample.KSample` for more information.

    metric : str or function, (default="gaussian")
        Distance or a kernel metric to use, either a callable or a valid string.
        If a callable, then it should behave similarly to either
        :func:`sklearn.metrics.pairwise_distances` or to
        :func:`sklearn.metrics.pairwise.pairwise_kernels`.
        If a string, then it should be either one of the keys in either
        `sklearn.metrics.pairwise.PAIRED_DISTANCES` or in
        `sklearn.metrics.pairwise.PAIRWISE_KERNEL_FUNCTIONS`, or "gaussian",
        which will use a gaussian kernel with an adaptively selected bandwidth.
        It is recommended to use kernels (e.g. "gaussian") with kernel-based
        hsic test and distances (e.g. "euclidean") with all other tests.

    n_components : int or None, optional (default=None)
        Number of embedding dimensions. If None, the optimal embedding
        dimensions are found by the Zhu and Godsi algorithm.
        See :func:`~graspy.embed.selectSVD` for more information.

    n_bootstraps : int (default=200)
        Number of bootstrap iterations for the backend hypothesis test.
        See :class:`hyppo.ksample.KSample` for more information.

    workers : int, optional (default=1)
        Number of workers to use. If more than 1, parallelizes the code.
        Supply -1 to use all cores available to the Process.

    size_correction: bool (default=True)
        Ignored when the two graphs have the same number of vertices. The test degrades
        in validity as the number of vertices of the two graphs diverge from each other,
        unless a correction is performed.
        If True, when the two graphs have different numbers of vertices, estimates
        the plug-in estimator for the variance and uses it to correct the
        embedding of the larger graph.
        If False, does not perform any modifications (not recommended).

    Attributes
    ----------
    null_distribution_ : ndarray, shape (n_bootstraps, )
        The distribution of T statistics generated under the null.

    sample_T_statistic_ : float
        The observed difference between the embedded latent positions of the two
        input graphs.

    p_value_ : float
        The overall p value from the test.

    References
    ----------
    .. [1] Tang, M., Athreya, A., Sussman, D. L., Lyzinski, V., & Priebe, C. E. (2017).
        "A nonparametric two-sample hypothesis testing problem for random graphs."
        Bernoulli, 23(3), 1599-1630.

    .. [2] Panda, S., Palaniappan, S., Xiong, J., Bridgeford, E., Mehta, R., Shen, C., & Vogelstein, J. (2019).
        "hyppo: A Comprehensive Multivariate Hypothesis Testing Python Package."
        arXiv:1907.02088.

    .. [3] Varjavand, B., Arroyo, J., Tang, M., Priebe, C., and Vogelstein, J. (2019).
       "Improving Power of 2-Sample Random Graph Tests with Applications in Connectomics"
       arXiv:1911.02741

    .. [4] Alyakin, A., Agterberg, J., Helm, H., Priebe, C. (2020)
       "Correcting a Nonparametric Two-sample Graph Hypothesis test for Differing Orders"
       TODO cite the arXiv whenever possible
    """

    def __init__(
        self,
        test="dcorr",
        metric="euclidean",
        n_components=None,
        n_bootstraps=200,
        workers=1,
        size_correction=True,
    ):

        if not isinstance(test, str):
            msg = "test must be a str, not {}".format(type(test))
            raise TypeError(msg)
        elif test not in _VALID_TESTS:
            msg = "Unknown test {}. Valid tests are {}".format(test, _VALID_TESTS)
            raise ValueError(msg)

        if not isinstance(metric, str) and not callable(metric):
            msg = "Metric must be str or callable, not {}".format(type(metric))
            raise TypeError(msg)
        elif metric not in _VALID_METRICS and not callable(metric):
            msg = "Unknown metric {}. Valid metrics are {}, or a callable".format(
                metric, _VALID_METRICS
            )
            raise ValueError(msg)

        if n_components is not None:
            if not isinstance(n_components, int):
                msg = "n_components must be an int, not {}.".format(type(n_components))
                raise TypeError(msg)

        if not isinstance(n_bootstraps, int):
            msg = "n_bootstraps must be an int, not {}".format(type(n_bootstraps))
            raise TypeError(msg)
        elif n_bootstraps < 0:
            msg = "{} is invalid number of bootstraps, must be non-negative"
            raise ValueError(msg.format(n_bootstraps))

        if not isinstance(workers, int):
            msg = "workers must be an int, not {}".format(type(workers))
            raise TypeError(msg)

        if not isinstance(size_correction, bool):
            msg = "size_correction must be a bool, not {}".format(type(size_correction))
            raise TypeError(msg)

        super().__init__(n_components=n_components)

        if callable(metric):
            metric_func = metric
        else:
            if metric in _VALID_DISTANCES:
                if test == "hsic":
                    msg = (
                        f"{test} is a kernel-based test, but {metric} "
                        "is a distance. results may not be optimal. it is "
                        "recomended to use either a different test or one of "
                        f"the kernels: {_VALID_KERNELS} as a metric."
                    )
                    warnings.warn(msg, UserWarning)

                def metric_func(X, Y=None, metric=metric, workers=None):
                    return pairwise_distances(X, Y, metric=metric, n_jobs=workers)

            elif metric == "gaussian":
                if test != "hsic":
                    msg = (
                        f"{test} is a distance-based test, but {metric} "
                        "is a kernel. results may not be optimal. it is "
                        "recomended to use either a hisc as a test or one of "
                        f"the distances: {_VALID_DISTANCES} as a metric."
                    )
                    warnings.warn(msg, UserWarning)
                metric_func = gaussian
            else:
                if test != "hsic":
                    msg = (
                        f"{test} is a distance-based test, but {metric} "
                        "is a kernel. results may not be optimal. it is "
                        "recomended to use either a hisc as a test or one of "
                        f"the distances: {_VALID_DISTANCES} as a metric."
                    )
                    warnings.warn(msg, UserWarning)

                def metric_func(X, Y=None, metric=metric, workers=None):
                    return pairwise_kernels(X, Y, metric=metric, n_jobs=workers)

        self.test = KSample(test, compute_distance=metric_func)
        self.n_bootstraps = n_bootstraps
        self.workers = workers
        self.size_correction = size_correction

    def _embed(self, A1, A2):
        if self.n_components is None:
            num_dims1 = select_dimension(A1)[0][-1]
            num_dims2 = select_dimension(A2)[0][-1]
            self.n_components = max(num_dims1, num_dims2)

        ase = AdjacencySpectralEmbed(n_components=self.n_components)
        X1_hat = ase.fit_transform(A1)
        X2_hat = ase.fit_transform(A2)

        if isinstance(X1_hat, tuple) and isinstance(X2_hat, tuple):
            X1_hat = np.concatenate(X1_hat, axis=-1)
            X2_hat = np.concatenate(X2_hat, axis=-1)
        elif isinstance(X1_hat, tuple) ^ isinstance(X2_hat, tuple):
            msg = (
                "input graphs do not have same directedness. "
                "consider symmetrizing the directed graph."
            )
            raise ValueError(msg)

        return X1_hat, X2_hat

    def _sample_modified_ase(self, X, Y, pooled=False):
        N, M = len(X), len(Y)

        # return if graphs are same order, else else ensure X the larger graph.
        if N == M:
            return X, Y
        elif M > N:
            reverse_order = True
            X, Y = Y, X
            N, M = M, N
        else:
            reverse_order = False

        # estimate the central limit theorem variance
        if pooled:
            # TODO unclear whether using pooled estimator provides more power.
            # TODO this should be investigated. should not matter under null.
            two_samples = np.concatenate([X, Y], axis=0)
            get_sigma = _fit_plug_in_variance_estimator(two_samples)
        else:
            get_sigma = _fit_plug_in_variance_estimator(X)
        X_sigmas = get_sigma(X) * (N - M) / (N * M)

        # increase the variance of X by sampling from the asy dist
        X_sampled = np.zeros(X.shape)
        # TODO may be parallelized, but requires keeping track of random state
        for i in range(N):
            X_sampled[i, :] = X[i, :] + stats.multivariate_normal.rvs(cov=X_sigmas[i])

        # return the embeddings in the appropriate order
        return (Y, X_sampled) if reverse_order else (X_sampled, Y)

    def fit(self, A1, A2):
        """
        Fits the test to the two input graphs

        Parameters
        ----------
        A1, A2 : nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.MultiGraph, np.ndarray
            The two graphs to run a hypothesis test on.

        Returns
        -------
        self
        """
        A1 = import_graph(A1)
        A2 = import_graph(A2)

        X1_hat, X2_hat = self._embed(A1, A2)
        X1_hat, X2_hat = _median_sign_flips(X1_hat, X2_hat)

        if self.size_correction:
            X1_hat, X2_hat = self._sample_modified_ase(X1_hat, X2_hat)

        data = self.test.test(
            X1_hat, X2_hat, reps=self.n_bootstraps, workers=self.workers, auto=False
        )

        self.null_distribution_ = self.test.indep_test.null_dist
        self.sample_T_statistic_ = data[0]
        self.p_value_ = data[1]

        return self


def _median_sign_flips(X1, X2):
    X1_medians = np.median(X1, axis=0)
    X2_medians = np.median(X2, axis=0)
    val = np.multiply(X1_medians, X2_medians)
    t = (val > 0) * 2 - 1
    X1 = np.multiply(t.reshape(-1, 1).T, X1)
    return X1, X2


def _fit_plug_in_variance_estimator(X):
    """
    Takes in ASE of a graph and returns a function that estimates
    the variance-covariance matrix at a given point using the
    plug-in estimator from the RDPG Central Limit Theorem.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        adjacency spectral embedding of a graph

    Returns
    -------
    plug_in_variance_estimtor: functions
        a function that estimates variance (see below)
    """

    n = len(X)

    # precompute the Delta and the middle term matrix part
    delta = 1 / (n) * (X.T @ X)
    delta_inverse = np.linalg.inv(delta)
    middle_term_matrix = np.einsum("bi,bo->bio", X, X)

    def plug_in_variance_estimator(x):
        """
        Takes in a point of a matrix of points in R^d and returns an
        estimated covariance matrix for each of the points

        Parameters:
        -----------
        x: np.ndarray, shape (n, d)
            points to estimate variance at
            if 1-dimensional - reshaped to (1, d)

        Returns:
        -------
        covariances: np.ndarray, shape (n, d, d)
            n estimated variance-covariance matrices of the points provided
        """
        if x.ndim < 2:
            x = x.reshape(1, -1)
        # the following two lines are a properly vectorized version of
        # middle_term = 0
        # for i in range(n):
        #     middle_term += np.multiply.outer((x @ X[i] - (x @ X[i]) ** 2),
        #                                      np.outer(X[i], X[i]))
        # where the matrix part does not involve x and has been computed above
        middle_term_scalar = x @ X.T - (x @ X.T) ** 2
        middle_term = np.tensordot(middle_term_scalar, middle_term_matrix, axes=1)
        covariances = delta_inverse @ (middle_term / n) @ delta_inverse
        return covariances

    return plug_in_variance_estimator


# %%
import time

np.random.seed(8888)
small_n_per_comm = [200, 200, 200]
big_n_per_comm = [300, 300, 300]
block_probs = np.array([[0.5, 0.1, 0.07], [0.1, 0.6, 0.05], [0.07, 0.05, 0.4]])

n_sims = 5
for i in range(n_sims):
    ldt = LatentDistributionTest(n_components=3, n_bootstraps=200, workers=-1)
    small_adj, small_labels = sbm(small_n_per_comm, block_probs, return_labels=True)
    big_adj, big_labels = sbm(big_n_per_comm, block_probs, return_labels=True)
    curr_time = time.time()
    ldt.fit(small_adj, big_adj)
    print(f"p value: {ldt.p_value_}")
    print(f"{(time.time() - curr_time)/60} min. elapsed.")

# %%
small_n_per_comm = [100, 100, 100]
big_n_per_comm = [400, 400, 400]
N = np.sum(big_n_per_comm)
M = np.sum(small_n_per_comm)
block_probs = np.array([[0.5, 0.1, 0.07], [0.1, 0.6, 0.05], [0.07, 0.05, 0.4]])
ldt = LatentDistributionTest(n_components=3, n_bootstraps=0, workers=-1)
small_adj, small_labels = sbm(small_n_per_comm, block_probs, return_labels=True)
big_adj, big_labels = sbm(big_n_per_comm, block_probs, return_labels=True)
X1_hat, X2_hat = ldt._embed(small_adj, big_adj)
X1_hat, X2_hat = _median_sign_flips(X1_hat, X2_hat)
X1_hat_corrected, X2_hat_corrected = ldt._sample_modified_ase(X1_hat, X2_hat)
get_sigma = _fit_plug_in_variance_estimator(X2_hat)
X_sigmas = get_sigma(X2_hat) * (N - M) / (N * M)

scatter_kws = dict(
    x=0,
    y=1,
    hue="label",
    palette=sns.color_palette("deep", len(np.unique(small_labels))),
    s=20,
    linewidth=0,
    alpha=0.5,
)

fig, axs = plt.subplots(2, 2, figsize=(16, 16))
axs = axs.ravel()

ax = axs[0]
ax.set_title("Smaller graph latent positions")
plot_embed = pd.DataFrame(data=X1_hat)
plot_embed["label"] = small_labels.astype("str")
sns.scatterplot(data=plot_embed, ax=ax, **scatter_kws)
ax.axis("off")
ax.get_legend().remove()

ax = axs[1]
ax.set_title("Larger graph latent positions")
plot_embed = pd.DataFrame(data=X2_hat)
plot_embed["label"] = big_labels.astype("str")
sns.scatterplot(data=plot_embed, ax=ax, **scatter_kws)
ax.axis("off")
ax.get_legend().remove()
# ax.legend(bbox_to_anchor=(1, 1,), loc="upper left")

ax = axs[2]
ax.set_title("Larger graph latent positions \n(w/ some estimated covariances)")
sns.scatterplot(data=plot_embed, ax=ax, **scatter_kws)
ax.axis("off")
ax.get_legend().remove()

n_ellipses = 20
choice_inds = np.random.choice(len(X2_hat), n_ellipses, replace=False)
for i, (x, cov) in enumerate(zip(X2_hat, X_sigmas)):
    if i in choice_inds:
        make_ellipse(x, cov, ax=ax, alpha=0.4, fill=False)

ax = axs[3]
ax.set_title("Larger graph latent positions \n(after resampling)")
plot_embed = pd.DataFrame(data=X2_hat_corrected)
plot_embed["label"] = big_labels.astype("str")
sns.scatterplot(data=plot_embed, ax=ax, **scatter_kws)
ax.axis("off")
ax.get_legend().remove()
plt.tight_layout()
plt.savefig("./ase_correction/ase_correction.png", fmt="png", dpi=300)

