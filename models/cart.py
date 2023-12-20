import numpy as np


# 将节点划分为左节点,构建二叉树
class Node:
    def __init__(self):
        # 需要记录当前的元素是连续还是离散的
        self.mem = None  # 用于分割的元素
        self.dimension = None  # 第几维度
        self.data = None  # 一个数组
        self.label = None
        self.target = None
        self.leftNode = None
        self.rightNode = None
        self.numOfLeaf = 0

    def predict(self, data):
        if self.leftNode == self.rightNode == None:
            return self.target
        tag = data[self.dimension]
        if type(tag) == str or type(tag) == bool:
            if tag == self.mem:
                return self.leftNode.predict(data)
            else:
                return self.rightNode.predict(data)
        else:
            if tag <= self.mem:
                return self.leftNode.predict(data)
            else:
                return self.rightNode.predict(data)

    def oneNodeCart(self, threshold=0.05):
        # 叶节点
        if np.sum(self.label) == len(self.label) or np.sum(self.label) == 0:
            count_true = np.sum(self.label)
            count_false = len(self.label) - count_true
            self.target = True if count_true > count_false else False
            return 1
        shape = self.data.shape
        print(self.data.shape)
        minGini = 1e9
        dimension = -1
        mem = -1
        for i in range(shape[1]):
            target_men, Gini = minimumOfOneAtttributeGini(self.data[:, i], self.label)
            if Gini < minGini:
                minGini = Gini
                dimension = i
                mem = target_men
        # 一个节点分列,需要分类为左节点和右节点,保留每个节点是按什么标准分裂的信息
        if not (minGini == 1e9 or minGini <= threshold):
            leftNode, rightNode = Node(), Node()
            if (
                type(self.data[:, dimension][0]) == str
                or type(self.data[:, dimension][0]) == bool
            ):
                leftNode.data = self.data[self.data[:, dimension] == mem]
                leftNode.label = self.label[self.data[:, dimension] == mem]
                rightNode.data = self.data[self.data[:, dimension] != mem]
                rightNode.label = self.label[self.data[:, dimension] != mem]
            else:
                leftNode.data = self.data[self.data[:, dimension] <= mem]
                leftNode.label = self.label[self.data[:, dimension] <= mem]
                rightNode.data = self.data[self.data[:, dimension] > mem]
                rightNode.label = self.label[self.data[:, dimension] > mem]
            self.leftNode = leftNode
            self.rightNode = rightNode
            self.dimension = dimension
            self.mem = mem
            # 递归下去
            numOfLeftLeaf = self.leftNode.oneNodeCart(threshold=threshold)
            numOfRightLeaf = self.rightNode.oneNodeCart(threshold=threshold)
            self.numOfLeaf = numOfLeftLeaf + numOfRightLeaf
            return self.numOfLeaf
        # 不需要再分类,停止操作
        count_true = np.sum(self.label)
        count_false = len(self.label) - count_true
        self.target = True if count_true > count_false else False
        return 1

    def pruning(self):
        # TODO:没有硬性要求
        ...


class CartTree:
    def __init__(self):
        self.root = Node()

    def fit(self, data, label, threshold=0.05):
        self.root.data = data
        self.root.label = label
        self.root.oneNodeCart(threshold)

    def predict(self, data):
        result = []
        for i in range(len(data)):
            result.append(self.root.predict(data[i]))
        return result


def Gini(data, label):
    #! 这种写法仅支持2分类
    p0 = np.count_nonzero(label == 0) / len(label)
    return 2 * p0 * (1 - p0)


# 一个类的基尼指数
def minimumOfOneAtttributeGini(data, label):
    """
    首先获取每个列的所有取值,排序后进行遍历,连续值取(a+b)/2,非连续值就取自己
    如果列已经只有单个元素就不再分了
    data是单列
    """
    # 当前维度只有一种数据,不需要再分了
    if len(np.unique(data)) == 1:
        return 1e9, 1e9
    minGini = 1e9
    target_men = 0
    if type(data[0]) == int or type(data[0]) == float:  # 连续值
        sorted_data = np.sort(np.unique(data))
        preMean = ""
        for i in range(len(sorted_data) - 1):
            mean = (sorted_data[i] + sorted_data[i + 1]) / 2
            if preMean == mean:
                # 降低复杂度
                continue
            preMean = mean
            dataLeft = data[data <= mean]
            labelLeft = label[data <= mean]

            dataRight = data[data > mean]
            labelRight = label[data > mean]
            gini = (
                len(dataLeft) * Gini(dataLeft, labelLeft)
                + len(dataRight) * Gini(dataRight, labelRight)
            ) / len(data)
            if minGini > gini:
                minGini = gini
                target_men = mean
    else:
        unique_arr = np.unique(data)
        for i in unique_arr:
            dataLeft = data[data == i]
            labelLeft = label[data == i]

            dataRight = data[data != i]
            labelRight = label[data != i]
            gini = (
                len(dataLeft) * Gini(dataLeft, labelLeft)
                + len(dataRight) * Gini(dataRight, labelRight)
            ) / len(data)
            if minGini > gini:
                minGini = gini
                target_men = i
    return (
        target_men,
        minGini,
    )
