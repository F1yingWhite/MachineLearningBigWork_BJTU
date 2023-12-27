import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics


def Myel(train, label, test):
    dtrain = xgb.DMatrix(train, label=label)
    params = {
        "max_depth": 5,
        "eta": 0.07,
        "objective": "binary:logistic",
    }
    dtest = xgb.DMatrix(test)
    # 训练模型
    model = xgb.train(params, dtrain, num_boost_round=20)
    # 在测试集上进行预测
    preds = model.predict(dtest)
    pred_labels = [True if x >= 0.5 else False for x in preds]
    return pred_labels


class GBC:
    def __init__(self):
        self.gbc = None

    def fit(self, data, label):
        gbc = GradientBoostingClassifier()
        gbc.fit(data, label)
        param_grid = {
            "learning_rate": [0.01, 0.1, 0.15, 0.2],
            "n_estimators": [30, 40, 50, 60],
            "max_depth": [3, 4, 5, 6],
        }
        grid_search = GridSearchCV(
            estimator=gbc, param_grid=param_grid, scoring="accuracy", cv=2
        )
        grid_search.fit(data, label)
        self.gbc = grid_search.best_estimator_

    def predict(self, data):
        return self.gbc.predict(data)
