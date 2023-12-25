import numpy as np
from models.cart import CartTree
import math
import random
import bisect


def random_sampling(probabilities, data, label):
    cum_probabilities = [sum(probabilities[: i + 1]) for i in range(len(probabilities))]
    samples = []
    n = len(data)
    for _ in range(n):
        r = random.random() * cum_probabilities[-1]
        # 二分查找概率分布数组中第一个大于 r 的位置
        idx = bisect.bisect_left(cum_probabilities, r)
        samples.append(idx)
    return data[samples], label[samples]


class AdaBoost:
    def __init__(self, num):
        self.trees = []
        self.alphas = []
        for i in range(num):
            self.trees.append(CartTree())

    def alpha(self, error):
        # 权重
        return 0.5 * math.log((1 - error) / error)

    def fit(self, data, label, threshold=0.05):
        weight = np.full(len(data), 1 / len(data), dtype=float)
        for i in range(len(self.trees)):
            # ? 如何训练一个带权重的基学习器,重采样么
            tempData, tempLabel = random_sampling(weight, data, label)
            self.trees[i].fit(tempData, tempLabel, threshold)
            error = self.errorRate(self.trees[i], data, label, weight)
            print(error)
            self.alphas.append(self.alpha(error))
            weight = self.updateWeight(
                self.trees[i], data, label, weight, self.alphas[i]
            )

    def updateWeight(self, tree, data, label, weight, alpha):
        newWeight = []
        tag = tree.predict(data)
        for i in range(len(tag)):
            temp = -1 if label[i] != tag[i] else 1
            newWeight.append(weight[i] * (math.e ** (-alpha * temp)))
        newWeight /= np.sum(newWeight)
        return newWeight

    def errorRate(self, tree, data, label, weight):
        error = 0
        result = tree.predict(data)
        for i in range(len(result)):
            if result[i] != label[i]:
                error += weight[i]
        return error

    def predict(self, data):
        result = np.zeros((len(self.trees), len(data)))
        for i in range(len(self.trees)):
            result[i] = self.trees[i].predict(data)
        weighted_array = result * np.array(self.alphas)[:, np.newaxis]
        # 每列相加
        result = weighted_array.sum(axis=0) / np.sum(np.array(self.alphas))
        return result > 0.5
