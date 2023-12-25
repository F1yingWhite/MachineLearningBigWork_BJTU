import numpy as np
from models.cart import CartTree
import math


class RandomForest:
    def __init__(self, num=6) -> None:
        self.trees = []
        self.featureIndex = []
        for i in range(num):
            self.trees.append(CartTree())

    def fit(self, data, label, choice=0.8, threshold=0.05):
        n = int(round(math.log(data.shape[1], 2)))
        new_col = label[:, np.newaxis]
        # 使用 np.concatenate 将数组拼接成一个新的二维数组
        new_data = np.concatenate((data, new_col), axis=1)
        for i in range(len(self.trees)):
            random_indices = np.random.choice(new_data.shape[0], int(len(new_data) * choice))
            data = new_data[random_indices]
            label = data[:, -1]
            data = data[:, :-1]
            cols = np.random.choice(data.shape[1], size=n, replace=False)
            selected_cols = data[:, cols]
            self.featureIndex.append(cols)
            self.trees[i].fit(selected_cols, label, threshold=threshold)
            print("一颗完成啦!")

    def predict(self, data):
        result = []
        for i in range(len(self.trees)):
            tempData = data[:, self.featureIndex[i]]
            result.append(self.trees[i].predict(tempData))
        result = np.array(result)
        # 统计每一列中的 0 的数量
        num_zeros = result.shape[0] - np.count_nonzero(result, axis=0)

        # 统计每一列中的 1 的数量
        num_ones = result.shape[0] - num_zeros
        return num_zeros < num_ones
