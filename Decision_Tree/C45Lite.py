# 参考博客https://cloud.tencent.com/developer/article/1057143

import sys
from InformationGain import *
sys.path.append('..')


class C45Lite:
    def __init__(self, D):
        self.D = D
        self.check = False
        self.mytree = self.create_tree(D)
        print(self.mytree)

    def majorityCnt(self, classList):
        p = {}
        for label in classList:
            if label in p:
                p[label] += 1
            else:
                p[label] = 1
        ma = None
        for key in p.keys():
            if ma is None or p[key] > p[ma]:
                ma = key

        return ma

    def chooseBestFeature(self, D):
        e = Ent(D, -1)
        idx = 0
        eu = Gain_utils(e, D, 0)
        for i in range(1, len(D[0]) - 1):
            cmp = Gain_utils(e, D, i)
            if cmp > eu:
                idx = i
                eu = cmp

        return idx

    def splitDataSet(self, D, a, v):
        n = []
        for row in D:
            if row[a] == v:
                n.append(np.delete(row, a))
        return np.array(n)

    def create_tree(self, D):
        classList = [label[-1] for label in D]
        if classList.count(classList[0]) == len(classList):
            return classList[0]
        if len(D[0]) == 1:
            return self.majorityCnt(classList)

        bestFeat = self.chooseBestFeature(D)
        mytree = {bestFeat: {}}
        featValues = [feat[bestFeat] for feat in D]
        featValues = set(featValues)
        for value in featValues:
            mytree[bestFeat][value] = self.create_tree(self.splitDataSet(D, bestFeat, value))

        return mytree

    def predict(self, D):
        for key in self.mytree.keys():
            pass
