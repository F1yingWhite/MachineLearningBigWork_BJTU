import pandas as pd
import numpy as np
from models.svm import SVMMethods
from sklearn.preprocessing import StandardScaler
from models.cart import CartTree
import csv
from models.RandomForest import RandomForest
from models.myel import Myel, GBC
from models.AdaBoost import AdaBoost
from collections import Counter


def outlierHandler(data, quantile=99):
    p99 = np.percentile(data, quantile)
    data = np.where(data > p99, p99, data)
    return data


# 在这里加载数据与进行数据预处理,包括数据标准化以及缺失值补全
def loadData(mode="number"):
    path = "./spaceship-titanic/train.csv"
    df = pd.read_csv(path)
    label = df["Transported"].replace(False, 0).replace(True, 1)
    path = "./spaceship-titanic/test.csv"
    dfTest = pd.read_csv(path)
    trainLen = df.__len__()
    df = pd.concat([df, dfTest])
    # 缺失值处理,年龄小于13和休眠的人没有开销
    df.loc[
        (df["CryoSleep"] == True) | (df["Age"] < 13),
        ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"],
    ] = 0

    # 乘客ID
    OriginalPassengerId = df["PassengerId"]  # 无空值,不过暂时不用
    PassengerId = [(s[:4]) for s in OriginalPassengerId]
    counter = Counter(PassengerId)
    OriginalPassengerId = OriginalPassengerId.factorize()
    sorted_counter = dict(sorted(counter.items()))
    GroupNum = [sorted_counter[s] for s in PassengerId]
    # 离开的星球,改为整数型
    HomePlanet = df["HomePlanet"].fillna(df["HomePlanet"].value_counts().idxmax())

    # 是否在休眠
    CryoSleep = df["CryoSleep"].fillna(df["CryoSleep"].value_counts().idxmax())
    # 仓位号,采用deck/num/side 的形式
    Cabin = df["Cabin"]  # .fillna(df["Cabin"].value_counts().idxmax())
    df[["CabinDeck", "CabinNo", "CabinSide"]] = Cabin.str.split("/", expand=True)
    CabinDeck = df["CabinDeck"].fillna(df["CabinDeck"].value_counts().idxmax())
    CabinSide = df["CabinSide"].fillna(df["CabinSide"].value_counts().idxmax())
    CabinNo = df["CabinNo"].fillna(df["CabinNo"].value_counts().idxmax())
    # 目的地
    Destination = df["Destination"].fillna(df["Destination"].value_counts().idxmax())
    # 年龄
    Age = df["Age"].fillna(df["Age"].mean())
    # 是否是VIP
    VIP = df["VIP"].fillna(df["VIP"].value_counts().idxmax())
    # 开支,反正没啥用,不如用了再说
    RoomService = df["RoomService"].fillna(df["RoomService"].mean())
    FoodCourt = df["FoodCourt"].fillna(df["FoodCourt"].mean())
    ShoppingMall = df["ShoppingMall"].fillna(df["ShoppingMall"].mean())
    Spa = df["Spa"].fillna(df["Spa"].mean())
    VRDeck = df["VRDeck"].fillna(df["VRDeck"].mean())
    # 新的3列
    NormalExpendtion = FoodCourt + ShoppingMall
    LuxuryExpendtion = Spa + VRDeck + RoomService
    TotalExpendtion = RoomService + FoodCourt + ShoppingMall + Spa + VRDeck
    if mode == "number":
        HomePlanet, _ = HomePlanet.factorize()
        CryoSleep, _ = CryoSleep.factorize()
        CabinDeck, _ = CabinDeck.factorize()
        CabinSide, _ = CabinSide.factorize()
        Destination, _ = Destination.factorize()
        VIP, _ = VIP.factorize()

    data = np.vstack(
        (
            # OriginalPassengerId,
            GroupNum,
            HomePlanet,
            CryoSleep,
            CabinDeck,
            # CabinNo,
            CabinSide,
            Destination,
            Age,
            VIP,
            RoomService,
            FoodCourt,
            ShoppingMall,
            Spa,
            VRDeck,
            NormalExpendtion,
            LuxuryExpendtion,
            TotalExpendtion,
        )
    ).T
    if mode == "number":
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
    return (
        data[:trainLen],
        np.array(label),
        data[trainLen:],
        dfTest["PassengerId"],
    )


def to_csv(predict, index, path):
    result = [["PassengerId", "Transported"]]
    for i in range(len(index)):
        data = [index[i], "True" if predict[i] else "False"]
        result.append(data)

    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(result)


if __name__ == "__main__":
    # train, label, test, index = loadData(mode="number")
    # SVMMethods(train, label, test, index)

    # train, label, test, index = loadData(mode="00")
    # tree = CartTree()
    # tree.fit(train, label, threshold=0.3)
    # to_csv(tree.predict(test), index, "result_cart.csv")

    # train, label, test, index = loadData(mode="00")
    # tree = RandomForest(num=23)
    # tree.fit(train, label, threshold=0.16)
    # to_csv(tree.predict(test), index, "result_rf.csv")

    # train, label, test, index = loadData(mode="00")
    # tree = AdaBoost(10)
    # tree.fit(train, label, threshold=0.2)
    # result = tree.predict(test)
    # to_csv(result, index, "result_ada.csv")

    # train, label, test, index = loadData(mode="number")
    # predict = Myel(train, label, test)
    # to_csv(predict, index, "result_myel.csv")

    train, label, test, index = loadData(mode="number")
    classifier = GBC()
    classifier.fit(train, label)
    to_csv(classifier.predict(test), index, "result_myel.csv")
