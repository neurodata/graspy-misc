#%%
import numpy as np
import seaborn as sns

from graspy.inference import LatentDistributionTest
from graspy.simulations import p_from_latent, sample_edges
from tqdm import tqdm

n_sims = 1000
n_verts = 200
n_components = 3
latent_size = (n_verts, n_components)
directed = False
latent = np.random.uniform(0.2, 0.5, size=latent_size)

p_mat = p_from_latent(latent, rescale=False, loops=False)

sim_p_vals = np.zeros(n_sims)
for i in tqdm(range(n_sims)):
    graph1 = sample_edges(p_mat, directed=directed, loops=False)
    graph2 = sample_edges(p_mat, directed=directed, loops=False)
    ldt = LatentDistributionTest(n_components=n_components, n_bootstraps=1000)
    out = ldt.fit(graph1, graph2)
    p_val = ldt.p_
    sim_p_vals[i] = p_val
#%%
from graspy.plot import pairplot

pairplot(latent)

from graspy.embed import AdjacencySpectralEmbed

ase = AdjacencySpectralEmbed(n_components=3)
latent_hat = ase.fit_transform(graph1)
pairplot(latent_hat)
latent_hat = ase.fit_transform(graph2)
pairplot(latent_hat)
#%%
sns.set_context("paper", font_scale=1.5)
plt.style.use("seaborn-white")
sns.set_palette("deep")
sns.distplot(sim_p_vals)
plt.xlabel("P-value")
plt.title(f"n_sims: {n_sims}, n_verts: {n_verts}, n_components: {n_components}")


# #%%
# from mgcpy.independence_tests.dcorr import DCorr
# from mgcpy.hypothesis_tests.transforms import k_sample_transform

# n_sims = 10000
# p_vals = np.zeros(n_sims)
# for i in tqdm(range(n_sims)):
#     latent1 = np.random.uniform(0.2, 0.7, size=latent_size)
#     latent2 = np.random.uniform(0.2, 0.7, size=latent_size)

#     sample, indicator = k_sample_transform(latent1, latent2)
#     test = DCorr("unbiased")
#     p, p_meta = test.p_value(sample, indicator, replication_factor=1000, is_fast=False)
#     p_vals[i] = p
# sns.distplot(p_vals)
# #%%


#%%
