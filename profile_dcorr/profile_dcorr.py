#%%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from dcor.independence import distance_covariance_test
from tqdm import tqdm

from mgcpy.hypothesis_tests.transforms import k_sample_transform
from mgcpy.independence_tests.dcorr import DCorr

n_sims = 500
n_verts = 200
n_components = 3
n_permutations = 1000
latent_size = (n_verts, n_components)
directed = False

p_vals = np.zeros(n_sims)
for i in tqdm(range(n_sims)):
    latent1 = np.random.uniform(0.2, 0.7, size=latent_size)
    latent2 = np.random.uniform(0.2, 0.7, size=latent_size)

    sample, indicator = k_sample_transform(latent1, latent2)
    test = DCorr("unbiased")
    p, p_meta = test.p_value(sample, indicator, replication_factor=1000, is_fast=False)
    p_vals[i] = p
plt.figure()
sns.distplot(p_vals)
plt.title("MGCPy DCorr, 2-sample under null")
plt.xlabel("p-value")
plt.savefig("graspy-misc/profile_dcorr/mgcpy_dcorr.png")

p_vals = np.zeros(n_sims)
for i in tqdm(range(n_sims)):
    latent1 = np.random.uniform(0.2, 0.7, size=latent_size)
    latent2 = np.random.uniform(0.2, 0.7, size=latent_size)
    sample, indicator = k_sample_transform(latent1, latent2)
    out = distance_covariance_test(sample, indicator, num_resamples=1000)
    p_vals[i] = out.p_value

plt.figure()
sns.distplot(p_vals)
plt.title("dcor, 2-sample under null")
plt.xlabel("p-value")
plt.savefig("graspy-misc/profile_dcorr/dcor.png")

