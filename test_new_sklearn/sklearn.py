# %% [markdown]
# #

from graspy.embed import AdjacencySpectralEmbed

from sklearn import __version__

print(__version__)

from sklearn.utils.estimator_checks import check_estimator

check_estimator(AdjacencySpectralEmbed)

