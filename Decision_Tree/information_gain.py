import numpy as np

def Ent(p):
    return -np.sum(p * np.log2(p))


def Gain_utils(pre, D, a, raw=1.0):
    t = {}
    for row in D:
        if row[a] in t:
            t[row[a]].append(row[-1])
        else:
            t[row[a]] = [row[-1]]

    gain = pre
    for key, val in t.items():
        p = {}
        for i in val:
            if i in p:
                p[i] += 1.0
            else:
                p[i] = 1.0
        p = np.array(list(p.values()))
        p /= len(val)
        gain -= Ent(p) * len(val) / len(D)
    gain *= raw
    V = []
    for i in t.values():
        V.append(len(i) / len(D))
    IV = Ent(V)
    gain_ratio = gain / IV

    return gain, gain_ratio

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

    gain, gain_ratio = Gain_utils(0.998, D, 0)
    print(gain, gain_ratio)