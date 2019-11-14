# %% [markdown]
# #
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from graspy.embed import AdjacencySpectralEmbed
from graspy.simulations import er_np

# Experiment parameters

n_verts = 200
p = 0.5
n_components = 1
n_sims = 1000


# Run experiment

estimated_latents = np.zeros((n_sims, 2))
for i in range(n_sims):
    graph = er_np(n_verts, p, directed=False, loops=False)

    ase_diag = AdjacencySpectralEmbed(n_components=n_components, diag_aug=True)

    ase = AdjacencySpectralEmbed(n_components=n_components, diag_aug=False)

    diag_latent = ase_diag.fit_transform(graph)
    ase_latent = ase.fit_transform(graph)

    mean_diag_latent = np.mean(diag_latent)
    mean_latent = np.mean(ase_latent)
    estimated_latents[i, 0] = mean_diag_latent
    estimated_latents[i, 1] = mean_latent

diffs = estimated_latents - np.sqrt(p)  # the true latent position is sqrt(p)


# Plot results

plt.style.use("seaborn-white")
sns.set_palette("deep")
sns.set_context("talk", font_scale=1)

plt.figure(figsize=(10, 5))
sns.distplot(diffs[:, 0], label="With diagaug")
sns.distplot(diffs[:, 1], label="Without diagaug")
plt.axvline(0, c="r", linestyle="--", label="True")
plt.xlabel("Difference from true latent position")
plt.legend()
plt.title(f"ER graphs, p = {p}, {n_verts} vertices, {n_sims} simulations")
plt.savefig(
    "./graspy-misc/diag-aug/diag-aug-er.png",
    fmt="png",
    dpi=150,
    facecolor="w",
    bbox_inches="tight",
    pad_inches=0.3,
)
