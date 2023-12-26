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
        self.myEl = None

    def fit(self, data, label):
        myEl = GradientBoostingClassifier(random_state=10)
        myEl.fit(data, label)
        param_grid = {
            "n_estimators": [30, 40, 50, 60, 70],
            "learning_rate": [0.01, 0.1, 0.15, 0.2, 0.25, 0.3],
            "max_depth": [3, 4, 5, 6, 10],
        }

        grid_search = GridSearchCV(
            estimator=myEl, param_grid=param_grid, scoring="accuracy", cv=3, n_jobs=-1
        )
        grid_search.fit(data, label)
        # 输出最佳参数和对应的准确度
        print("Best Parameters: ", grid_search.best_params_)
        print("Best Accuracy: %.4g" % grid_search.best_score_)
        # 使用最佳参数的模型在测试集上进行评估
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(data)
        accuracy = metrics.accuracy_score(label, y_pred)
        self.myEl = best_model
        return accuracy

    def predict(self, data):
        predict = self.myEl.predict(data)
        return predict
