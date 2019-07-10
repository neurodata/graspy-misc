#%%
from os.path import basename
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from graspy.inference import LatentDistributionTest
from graspy.simulations import sbm
from graspy.utils import symmetrize
from pandas import DataFrame
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# get where we are just to save output figure
folderpath = Path(__file__.replace(basename(__file__), ""))
savepath = folderpath / "outputs"

np.random.seed(8888)

B = [[0.5, 0.2], [0.2, 0.05]]
B = symmetrize(B)
k = 2
tests = 100
start = 50
stop = 800
diff = 25
alpha = 0.05
ns = []
ms = []
newms = []
error_list = []
temp = []


def fit(seed):
    warnings.filterwarnings("ignore")
    np.random.seed(seed)
    ldt = LatentDistributionTest(n_components=2, method="dcorr")
    p = ldt.fit(A1, A2)
    return p


for n in range(start, stop, diff):
    ns.append(n)
    for m in range(n, n + stop - start, diff):
        print(f"Running tests for n={n}, m={m}")
        cn = [n // k] * k
        cm = [m // k] * k
        A1 = sbm(cn, B)
        A2 = sbm(cm, B)
        type1_errors = 0

        seeds = np.random.randint(0, 1e8, tests)
        for p in range(tests):
            out = Parallel(n_jobs=-2, verbose=0)(delayed(fit)(seed) for seed in seeds)
            out = np.array(out)

            type1_errors += len(np.where(out < alpha)[0])

        error = type1_errors / tests
        temp.append(error)
        ms.append(m - n)
    error_list.append(temp)
    temp = []

for num in ms:
    if num not in newms:
        newms.append(num)

df = DataFrame(error_list, index=ns, columns=newms)
plt.figure(figsize=(15, 10))
sns.set_context("talk")
sns.heatmap(
    df, annot=True, linewidth=0.5, cmap="Reds", square=True, vmin=0, vmax=1, cbar=True
)
plt.title("Variation of Type 1 Error with Different Cases of LDT")
plt.xlabel("m - n")
plt.ylabel("n")
plt.savefig(savepath / "ldt_nm_scaling_sweep.pdf", format="pdf", facecolor="w")

