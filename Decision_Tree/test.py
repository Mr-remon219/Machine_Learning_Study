from sklearn.datasets import make_blobs
from sklearn.preprocessing import KBinsDiscretizer
import sys
from C45Lite import C45Lite
import numpy as np
sys.path.append("..")


if __name__ == '__main__':
    x, y = make_blobs(n_samples=20, n_features=5, centers=2)
    x = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform').fit_transform(x)
    D = np.hstack((x, y.reshape(-1, 1)))
    tree = C45Lite(D)

