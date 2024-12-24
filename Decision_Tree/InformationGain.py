import numpy as np

def Ent(D, l):
    p = {}
    for row in D:
        if row[l] in p:
            p[row[l]] += 1.0
        else:
            p[row[l]] = 1.0
    p = np.array(list(p.values()))
    p /= len(D)
    return -np.sum(p * np.log2(p))

def Gain_utils(ent, D, a, rho=1.0, dec=None):
    c = {}
    for row in D:
        if row[a] in c:
            c[row[a]].append(row)
        else:
            c[row[a]] = [row]

    gain = ent
    num = []
    for value in c.values():
        e = Ent(value, -1)
        gain -= e * len(value) / len(D)
        num.append(len(value) / len(D))
    gain *= rho
    IV = -np.sum(num * np.log2(num))
    gain_ratio = gain / IV

    if dec is not None:
        gain = round(gain, dec)
        gain_ratio = round(gain_ratio, dec)

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

    print(Gain_utils(Ent(D, 1), D, 0, dec=3))

