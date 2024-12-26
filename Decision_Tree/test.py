from sklearn.datasets import make_blobs
from sklearn.preprocessing import KBinsDiscretizer
import sys
from C45Lite import C45Lite
import numpy as np
sys.path.append("..")


if __name__ == '__main__':
    x, y = make_blobs(n_samples=10000, n_features=50, centers=4)
    x = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform').fit_transform(x)
    D = np.hstack((x.astype(int), y.reshape(-1, 1)))
    tree = C45Lite(D)

    z, Y = make_blobs(n_samples=1, n_features=50, centers=4)
    z = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform').fit_transform(x)
    res = tree.predict(z.astype(int))
    print(tree.get_tree())
    print(res)
