from graspy.datasets import load_drosophila_left, load_drosophila_right
from graspy.inference import LatentDistributionTest
left_graph = load_drosophila_left()
right_graph = load_drosophila_right()
ldt = LatentDistributionTest(n_components=3, n_bootstraps=500)
p_value = ldt.fit(left_graph, right_graph)
print("p-value: " + str(p_value))
