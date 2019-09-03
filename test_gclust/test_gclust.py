#%%
from graspy.cluster import GaussianCluster
from graspy.plot import pairplot
import numpy as np

n = 100
d = 3

np.random.seed(3)

X1 = np.random.normal(0.5, 0.5, size=(n, d))
X2 = np.random.normal(-0.5, 0.5, size=(n, d))
X3 = np.random.normal(0.8, 0.6, size=(n, d))
X4 = np.random.uniform(0.2, 0.3, size=(n, d))
X = np.vstack((X1, X2, X3, X4))
pairplot(X)

np.random.seed(3)

gclust = GaussianCluster(min_components=2, max_components=2, n_init=1, max_iter=100)
gclust.fit(X)

bic1 = gclust.bic_

np.random.seed(3)

gclust = GaussianCluster(min_components=2, max_components=2, n_init=50, max_iter=100)
gclust.fit(X)

bic2 = gclust.bic_

# we'd hope that bic2 is a little bit lower than bic1
print(bic1)
print(bic2)

