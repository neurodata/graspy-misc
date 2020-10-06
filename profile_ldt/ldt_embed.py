#%%
from graspy.inference import LatentDistributionTest

ldt = LatentDistributionTest(input_graph=False)


class Bob:
    pass


ldt.fit(Bob(), Bob())
