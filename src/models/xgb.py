import xgboost as xgb


def xgb_train(config, trn_x, trn_y, val_x, val_y, test):
    # prepare the xgboost dataset
    trn_data = xgb.DMatrix(trn_x, label=trn_y)
    val_data = xgb.DMatrix(val_x, label=val_y)

    # create the model
    model = xgb.train(
        params=config['params'],
        dtrain=trn_data,
        num_boost_round=config['num_boost_round'],
        evals=[(val_data, 'val')],
        early_stopping_rounds=config['early_stopping_rounds'],
        verbose_eval=100,
    )

    res = {}
    res['val_pred'] = model.predict(val_x)
    res['test_pred'] = model.predict(test.drop(['ts', 'target'], axis=1))
    return res
