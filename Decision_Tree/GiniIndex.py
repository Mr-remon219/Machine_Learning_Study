import numpy as np


def gini(D, l):
    p = {}
    for row in D:
        if row[l] in p:
            p[row[l]] += 1 / len(D)
        else:
            p[row[l]] = 1 / len(D)
    p = np.array(list(p.values()))
    return 1 - np.sum(p * p)

def gini_utils(D, a, dec=None):
    c = {}
    for row in D:
        if row[a] in c:
            c[row[a]].append(row)
        else:
            c[row[a]] = [row]

    gini_index = 0
    for value in c.values():
        gini_index += gini(value, -1) * len(value) / len(D)

    if dec is not None:
        gini_index = round(gini_index, dec)

    return gini_index


if __name__ == '__main__':
    D = [
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 1],
        [2, 1],
        [2, 1],
        [2, 1],
        [2, 0],
        [3, 0],
        [3, 0],
        [1, 0],
        [2, 0],
        [2, 0],
        [2, 0],
        [1, 0],
        [1, 0]
    ]

    print(gini_utils(D, 0, dec=3))
