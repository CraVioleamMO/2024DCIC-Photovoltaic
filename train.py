import pandas as pd
import numpy as np
from datetime import date, timedelta
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor
import lightgbm as lgb
from lightgbm import log_evaluation, early_stopping
import os
import warnings
import yaml
import json
import argparse
from tqdm import *
from src.models import lgbm_train, cat_train, xgb_train

warnings.filterwarnings('ignore')


def cv_model(model: dict, train, test, seed=2024):
    # set the vec to store the result
    cv_scores = []
    test_pred = np.zeros(test.shape[0])
    val_pred = np.zeros(train.shape[0])

    # create the fold
    kf = KFold(n_splits=5, random_state=seed, shuffle=True)

    for i, (trn_index, val_index) in enumerate(kf.split(train)):
        print('the %dth fold' % (i + 1))

        # split the data
        trn_x = train.iloc[trn_index].drop(['target', 'ts'], axis=1)
        trn_y = train.iloc[trn_index]['target']
        val_x = train.iloc[val_index].drop(['target', 'ts'], axis=1)
        val_y = train.iloc[val_index]['target']

        # train the model4
        result = {}
        if model['name'] == 'lgbm':
            result = lgbm_train(model['config'], trn_x, trn_y, val_x, val_y, test)
        elif model['name'] == 'cat':
            result = cat_train(model['config'], trn_x, trn_y, val_x, val_y, test)
        elif model['name'] == 'xgb':
            pass
        else:
            raise ValueError('The model name is not supported, the support model is [ lgbm, cat, xgb ]')

        val_pred[val_index] = result['val_pred']
        test_pred += result['test_pred'] / 5

        # count the RMSE
        cv_scores.append(np.sqrt(np.mean((val_pred[val_index] - val_y) ** 2)))

    print('cv scores:', cv_scores)
    # count the RMSE between the val_pred and the real target
    print('cv score:', np.sqrt(np.mean((val_pred - train['target']) ** 2)))
    return test_pred, val_pred


def split_by_day(df):
    # split the data into day_df and night_df
    day_df = df[df.half_day == 1].drop(['half_day'], axis=1)
    night_df = df[df.half_day == 0].drop(['half_day'], axis=1)
    return [day_df, night_df]


def split_by_user(df):
    day_df, night_df = split_by_day(df)
    if config['model']['name'] != 'cat':
        user_list = [5, 6, 8]
    else:
        user_list = ['f6', 'f7', 'f9']
    return [
        day_df[~day_df['UserID'].isin(user_list)],
        day_df[day_df['UserID'].isin(user_list)],
        night_df[~night_df['UserID'].isin(user_list)],
        night_df[night_df['UserID'].isin(user_list)]
    ]


def generate_submission(test, test_list, test_pred_list):
    submission = test[['UserID', 'ComRate', 'ts', 'half_day']]

    for tst, temp in zip(test_list, test_pred_list):
        # 保留四位小数
        temp = np.round(temp, 4)
        submission.loc[tst.index, 'target'] = temp
    submission = submission.drop(['half_day'], axis=1)

    # change the minus value to zero
    submission['target'] = submission['target'].apply(lambda x: 0 if x < 0 else x)
    # transform the ts to the date and not fill the zero when the day and month is less than 10
    submission['ts'] = submission['ts'].apply(lambda x: x.date())
    submission['p'] = test[['15minute_in_day']] + 1
    submission = submission.set_index(['UserID', 'ComRate', 'ts', 'p']).unstack()
    submission.columns = ['p' + str(i) for i in range(1, 97)]
    submission = submission.reset_index()
    submission = submission.rename(
        {'UserID': '光伏用户编号', 'ComRate': '综合倍率', 'ts': '时间'}, axis=1
    )
    submission['时间'] = submission['时间'].apply(lambda x: x.strftime('%Y-%#m-%#d 0:00'))

    return submission


if __name__ == '__main__':
    # read the config file from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/lgbm.yaml')
    args = parser.parse_args()

    # load the configuration file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print('The config file is')
    print(json.dumps(config, indent=4))

    # load the data
    df = pd.read_csv(config['input'], parse_dates=['ts'])

    # weather use extra data
    if config['extra_data']:
        extra_df = pd.read_csv(config['extra_data_path'], parse_dates=['ts'])
        df = df.merge(extra_df.drop(['ts'], axis=1), on=['UserID', 'year', 'month', 'day', 'hour'], how='left')

    if config['model']['name'] != 'cat':
        df['UserID'] = df['UserID'].astype('category').cat.codes

    # split the data into train and test
    train = df[df.is_test == False].drop(['is_test'], axis=1)
    test = df[df.is_test == True].drop(['is_test'], axis=1)

    # split the data into day_df and night_df
    train_list = []
    test_list = []

    if config['split'] == 'normal':
        train_list.append(train)
        test_list.append(test)
    elif config['split'] == 'half_day':
        train_list = split_by_day(train)
        test_list = split_by_day(test)
    elif config['split'] == 'user':
        train_list = split_by_user(train)
        test_list = split_by_user(test)

    # use label encoder to encode the UserID
    if config['model']['name'] != 'cat':
        df['UserID'] = df['UserID'].astype('category').cat.codes

    # start the model training
    test_pred_list = []
    val_pred_list = []

    # train the model
    for i in range(len(train_list)):
        print('The %dth data' % (i + 1))
        test_pred, val_pred = cv_model(config['model'], train_list[i], test_list[i])
        test_pred_list.append(test_pred)
        val_pred_list.append(val_pred)

    # generate the submission file
    submission = generate_submission(test, test_list, test_pred_list)

    # revise the UserID
    if config['model']['name'] != 'cat':
        submission['光伏用户编号'] = submission['光伏用户编号'].map(
            dict(zip([i for i in range(9)], ['f' + str(i) for i in range(1, 10)]))
        )

    # save the file
    submission.to_csv(config['output'], index=False)
