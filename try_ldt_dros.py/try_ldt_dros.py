from graspy.datasets import load_drosophila_left, load_drosophila_right
from graspy.inference import LatentDistributionTest

left_graph = load_drosophila_left()
right_graph = load_drosophila_right()

ldt = LatentDistributionTest(n_components=3, n_bootstraps=500)

ldt.fit(left_graph, right_graph)

p_value = ldt.p_

print(f"p-value from latent distribution test: {p_value}")
