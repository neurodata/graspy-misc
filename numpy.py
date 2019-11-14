#%%
import numpy as np
from graspy.simulations import sample_edges, er_np
from graspy.plot import heatmap


g = er_np(10, 0.5)
heatmap(g)
P = 0.5 * np.ones((10, 10))
g = sample_edges(P)
heatmap(g)
#%%
g == 1
P[g == 1] = 100
P[g == 0] = -100
P
heatmap(g)
heatmap(P)
# %%
directed = True
if directed:
    sample_edges(P, directed=True)
else:
    sample_edges(P, directed=False)

sample_edges(P, directed=directed)
# %%
def sample(P, directed=False):
    print(directed)
    print(P)
    G = sample_edges(P, directed=directed)
    return G


from graspy.utils import is_symmetric

sample(P)

# %%
