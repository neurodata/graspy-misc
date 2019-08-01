#%%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from dcor.independence import distance_covariance_test
from tqdm import tqdm

from mgcpy.hypothesis_tests.transforms import k_sample_transform
from mgcpy.independence_tests.dcorr import DCorr

np.random.seed(8888)
plt.style.use("seaborn-white")
sns.set_palette("deep")

n_sims = 10000
n_verts = 100
n_components = 2
n_permutations = 1000
size = (n_verts, n_components)
directed = False

#%% mgcpy package
p_vals = np.zeros(n_sims)
for i in tqdm(range(n_sims)):
    sample1 = np.random.uniform(0.2, 0.7, size=size)
    sample2 = np.random.uniform(0.2, 0.7, size=size)

    sample, indicator = k_sample_transform(sample1, sample2)
    test = DCorr(which_test="biased")
    p, p_meta = test.p_value(
        sample, indicator, replication_factor=n_permutations, is_fast=False
    )
    p_vals[i] = p

plt.figure()
sns.distplot(p_vals)
plt.title("MGCPy DCorr, 2-sample under null")
plt.xlabel("p-value")
plt.savefig("graspy-misc/profile_dcorr/mgcpy_dcorr.png", facecolor="w")

#%% dcor package
p_vals = np.zeros(n_sims)
for i in tqdm(range(n_sims)):
    sample1 = np.random.uniform(0.2, 0.7, size=size)
    sample2 = np.random.uniform(0.2, 0.7, size=size)
    sample, indicator = k_sample_transform(sample1, sample2)
    out = distance_covariance_test(sample, indicator, num_resamples=n_permutations)
    p_vals[i] = out.p_value

plt.figure()
sns.distplot(p_vals)
plt.title("dcor, 2-sample under null")
plt.xlabel("p-value")
plt.savefig("graspy-misc/profile_dcorr/dcor.png", facecolor="w")

