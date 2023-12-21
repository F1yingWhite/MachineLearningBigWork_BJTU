from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
import csv


def SVMMethods(train, label, test, index):
    clf = svm.SVC()
    clf.fit(train, label)
    predict = clf.predict(test)
    result = [["PassengerId", "Transported"]]
    for i in range(len(index)):
        data = [index[i], "True" if predict[i] else "False"]
        result.append(data)

    with open("result_svm.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(result)
