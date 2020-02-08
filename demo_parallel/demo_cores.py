import numpy as np
from joblib import Parallel, delayed


def _get_big_number():
    big_number = np.random.randint(1e8)
    return big_number


if __name__ == "__main__":
    n_numbers = 10000000
    np.random.seed(8888)
    par = Parallel(n_jobs=4)
    outs = par(delayed(_get_big_number)() for _ in range(n_numbers))
