import catboost as cat


def cat_train(config, trn_x, trn_y, val_x, val_y, test):
    # prepare the catboost dataset
    trn_data = cat.Pool(trn_x, label=trn_y,cat_features=config['cat_features'])
    val_data = cat.Pool(val_x, label=val_y,cat_features=config['cat_features'])

    # create the model
    model = cat.train(
        pool=trn_data,
        eval_set=[val_data],
        iterations=config['iterations'],
        params=config['params'],
        verbose=100,
        early_stopping_rounds=config['early_stopping_rounds'],
    )

    res = {}
    res['val_pred'] = model.predict(val_x)
    res['test_pred'] = model.predict(test.drop(['ts','target'],axis=1))
    return res
