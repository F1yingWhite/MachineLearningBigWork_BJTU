import xgboost as xgb


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
