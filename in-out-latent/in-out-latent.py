#%%
import graspy as gs
import numpy as np
import seaborn as sns

B = np.array([[0.3, 0.1, 0.1], [0.3, 0.1, 0.1], [0.1, 0.5, 0.1]])
sns.heatmap(B, square=True, annot=True, cmap="RdBu_r", center=0)
adj, labels = gs.simulations.sbm([100, 100, 100], B, return_labels=True, directed=True)
ase = gs.embed.AdjacencySpectralEmbed(diag_aug=False, n_components=3)
out_latent, in_latent = ase.fit_transform(adj)

gs.plot.pairplot(out_latent, labels=labels)
gs.plot.pairplot(in_latent, labels=labels)