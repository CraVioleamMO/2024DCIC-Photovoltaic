import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

def lgbm_train(config, trn_x, trn_y, val_x, val_y, test) -> dict:
    # prepare the lgb dataset
    trn_data = lgb.Dataset(trn_x, label=trn_y)
    val_data = lgb.Dataset(val_x, label=val_y)

    # start the training
    model = lgb.train(
        params=config['params'],
        train_set=trn_data,
        valid_sets=[trn_data, val_data],
        num_boost_round=config['num_boost_round'],
        callbacks=[log_evaluation(100), early_stopping(100)]
    )
    res = {}
    res['val_pred'] = model.predict(val_x, num_iteration=model.best_iteration)
    res['test_pred'] = model.predict(test.drop(['target', 'ts'], axis=1), num_iteration=model.best_iteration)
    return res
