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


def cv_model_plus(model: dict, train, test, seed=2024):
    # set the vec to store the result
    cv_scores = []
    test_pred = np.zeros(test.shape[0])

    # get data which the month=5,6,7 in train, but the B test' months are in [8,9,10,11,12]
    train_part = train[train.month.isin([8,9,10,11,12])]
    val_pred = np.zeros(train_part.shape[0])

    # create the fold
    kf = KFold(n_splits=5, random_state=seed, shuffle=True)

    for i, (trn_index, val_index) in enumerate(kf.split(train_part)):
        print('the %d fold' % (i + 1))

        # split the data
        val_x = train_part.iloc[val_index].drop(['target', 'ts'], axis=1)
        val_y = train_part.iloc[val_index]['target']

        # create the index
        index = np.ones(train.shape[0]).astype('bool')
        index[val_index] = False
        trn_x = train[index].drop(['target', 'ts'], axis=1)
        trn_y = train[index]['target']

        # train the model
        result = {}
        if model['name'] == 'lgbm':
            result = lgbm_train(model['config'], trn_x, trn_y, val_x, val_y, test)
        elif model['name'] == 'cat':
            result = cat_train(model['config'], trn_x, trn_y, val_x, val_y, test)
        elif model['name'] == 'xgb':
            result = xgb_train(model['config'], trn_x, trn_y, val_x, val_y, test)
        else:
            raise ValueError('The model name is not supported,the support model is [ lgbm, cat, xgb ]')

        val_pred[val_index] = result['val_pred']
        test_pred += result['test_pred'] / 5

        # count the RMSE
        cv_scores.append(np.sqrt(np.mean((val_pred[val_index] - val_y) ** 2)))

    print('cv scores:', cv_scores)
    # count the RMSE between the val_pred and the real target
    print('cv score:', np.sqrt(np.mean((val_pred - train_part['target']) ** 2)))
    return test_pred, val_pred


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
            result = xgb_train(model['config'], trn_x, trn_y, val_x, val_y, test)
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

    # filter the data by month
    if config.get('filter', False):
        # ensure the filter=[5,6,7] at least
        df = df[df.month.isin(config['filter'])]

    # revise the target which is less than 0 to 0
    if config.get('non_negative', False):
        df['target'] = df['target'].apply(lambda x: 0 if x < 0 else x)

    # weather use extra data
    if config['extra_data']:
        extra_df = pd.read_csv(config['extra_data_path'], parse_dates=['ts'])
        df = df.merge(extra_df.drop(['ts'], axis=1), on=['UserID', 'year', 'month', 'day', 'hour'], how='left')

    if config['model']['name'] != 'cat':
        df['UserID'] = df['UserID'].astype('category').cat.codes

    print('The shape of data is:', df.shape)
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

        if config.get('cv_plus', False):
            test_pred, val_pred = cv_model_plus(config['model'], train_list[i], test_list[i])
        else:
            test_pred, val_pred = cv_model(config['model'], train_list[i], test_list[i])
        test_pred_list.append(test_pred)
        val_pred_list.append(val_pred)

    # construct the val prediction result
    if config.get('val_output', False):
        train['val_pred'] = np.zeros(train.shape[0])
        result = None
        if config['cv_plus']:
            # train_part = train[train.month.isin([5, 6, 7])]
            train_part = train[train.month.isin([8,9,10,11,12])]
            for i in range(len(train_list)):
                # index = train_list[i][train_list[i].month.isin([5, 6, 7])].index
                index = train_list[i][train_list[i].month.isin([8,9,10,11,12])].index
                train_part.loc[index, 'val_pred'] = val_pred_list[i]
            result = train_part[['UserID', 'ts', 'target', 'val_pred']]
        else:
            for i in range(len(train_list)):
                train.loc[train_list[i].index, 'val_pred'] = val_pred_list[i]
            result = train[['UserID', 'ts', 'target', 'val_pred']]
        result.to_csv(config['val_output'], index=False)
    # generate the submission file
    submission = generate_submission(test, test_list, test_pred_list)

    # revise the UserID
    if config['model']['name'] != 'cat':
        submission['光伏用户编号'] = submission['光伏用户编号'].map(
            dict(zip([i for i in range(9)], ['f' + str(i) for i in range(1, 10)]))
        )

    # save the file
    submission.to_csv(config['output'], index=False)
