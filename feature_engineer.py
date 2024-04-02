# import the necessary libraries
import pandas as pd
import numpy as np
import polars as pl
from datetime import date, timedelta
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor
import os
import warnings
import yaml
import json
import time
import argparse
import logging
from tqdm import *

warnings.filterwarnings('ignore')


# define the decorator to calculate the time of function, and return the result of initial function
def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f'{func.__name__} cost time: {time.time() - start:.2f}s')
        return result

    return wrapper
# get the config file from the command line
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./config/fe.yaml', help='the path of config file')
args = parser.parse_args()

# use the config file to get the data path
with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

# use json to show the data
print('The config file is:')
print(json.dumps(config, indent=4))

'''
    Load Dataset
'''
# get the train and test data
print('**********load data**********')

path = os.path.join(config['input'], 'train')

train_basic = pd.read_csv(os.path.join(path, 'basic_information.csv'), encoding='ANSI')
train_weather = pd.read_csv(os.path.join(path, 'weather.csv'), encoding='ANSI', parse_dates=['时间'])
train_power = pd.read_csv(os.path.join(path, 'power.csv'), encoding='ANSI', parse_dates=['时间'])

path = os.path.join(config['input'], 'test')

test_basic = pd.read_csv(os.path.join(path, 'basic_information.csv'), encoding='ANSI')
test_weather = pd.read_csv(os.path.join(path, 'weather.csv'), encoding='ANSI', parse_dates=['时间'])
test_power = pd.read_csv(os.path.join(path, 'power.csv'), encoding='ANSI', parse_dates=['时间'])

print('**********data loaded**********')

'''
    Data process
'''
print('**********data process**********')
# add the p columns to test
p_cols = ['p' + str(i) for i in range(1, 97)]
test_power[p_cols] = np.zeros(shape=(test_power.shape[0], 96))


# unstack the data for expanding the data
@timeit
def preprocess_power(df: pd.DataFrame):
    temp = df.set_index(['光伏用户编号', '综合倍率', '时间']).stack().reset_index().rename(
        {'level_3': 'p', 0: 'target'}, axis=1)
    temp['时间'] = temp.apply(lambda row: row['时间'] + timedelta(minutes=15 * (int(row['p'][1:]) - 1)), axis=1)
    return temp.drop(['p'], axis=1)


train_power = preprocess_power(train_power)
test_power = preprocess_power(test_power)

# merge other information
train = train_power.merge(train_basic.drop(['光伏用户名称'], axis=1), on='光伏用户编号').merge(train_weather,
                                                                                               on=['光伏用户编号',
                                                                                                   '时间'])
test = test_power.merge(test_basic.drop(['光伏用户名称'], axis=1), on='光伏用户编号').merge(test_weather,
                                                                                            on=['光伏用户编号', '时间'])

# rename the columns
print('The initial name of columns is:')
print(train.columns)
columns = ['UserID', 'ComRate', 'ts', 'target', 'Capacity', 'Longitude', 'Latitude', 'AirPressure', 'RelativeHumidity',
           'CloudAmount', '10mWindSpeed', '10mWindDirection', 'Temperature', 'Radiation', 'Precipitation',
           '100mWindSpeed', '100mWindDirection']
train.columns = columns
test.columns = columns

print('The new name of columns is:')
print(train.columns)

'''
    Discard some data in train dataset
'''

start_month, end_month = config['discard']['start_month'], config['discard']['end_month']
train = train[(train['ts'].dt.month >= start_month) & (train['ts'].dt.month <= end_month)]

'''
    Feature Engineering
'''
# merge the train and test data
train['is_test'] = False
test['is_test'] = True
df = pd.concat([train, test], axis=0).reset_index(drop=True)

# sort by the User and ts
df = df.sort_values(['UserID', 'ts']).reset_index(drop=True)

# TODO revise the outliers

# transform the df to polar df
dfpl = pl.from_pandas(df)

'''
    1. Add time features
'''


@timeit
def get_time_feature(df: pl.DataFrame):
    # ordinal time feature
    df = df.with_columns(
        pl.col('ts').dt.year().alias('year'),
        pl.col('ts').dt.month().alias('month'),
        pl.col('ts').dt.day().alias('day'),
        pl.col('ts').dt.hour().alias('hour'),
        ((pl.col('ts').dt.hour() * 4 + pl.col('ts').dt.minute() / 15).cast(pl.Int16)).alias('15minute_in_day'),
        pl.col('ts').dt.ordinal_day().alias('day_of_year'),
        pl.col('ts').dt.weekday().alias('weekday'),
        (((pl.col('ts').dt.hour() + 6) % 24) // 12).cast(pl.Int16).alias('half_day'),
        (pl.col('ts').dt.hour() // 6).cast(pl.Int16).alias('6hour'),
    )
    # the fourier time feature
    df = df.with_columns(
        (2 * np.pi * pl.col('hour').sin() / 24).alias('hour_sin').cast(pl.Float32),
        (2 * np.pi * pl.col('hour') / 24).sin().alias('sin_hour_rad').cast(pl.Float32),
        (2 * np.pi * pl.col('hour').cos() / 24).alias('hour_cos').cast(pl.Float32),
        (2 * np.pi * pl.col('hour') / 24).cos().alias('cos_hour_rad').cast(pl.Float32),
        (2 * np.pi * pl.col('15minute_in_day').sin() / 96).alias('15minute_in_day_sin').cast(pl.Float32),
        (2 * np.pi * pl.col('15minute_in_day') / 96).sin().alias('sin_15minute_in_day_rad').cast(pl.Float32),
        (2 * np.pi * pl.col('15minute_in_day').cos() / 96).alias('15minute_in_day_cos').cast(pl.Float32),
        (2 * np.pi * pl.col('15minute_in_day') / 96).cos().alias('cos_15minute_in_day_rad').cast(pl.Float32),
        (2 * np.pi * pl.col('day_of_year').sin() / 365).alias('day_of_year_sin').cast(pl.Float32),
        (2 * np.pi * pl.col('day_of_year') / 365).sin().alias('sin_day_of_year_rad').cast(pl.Float32),
        (2 * np.pi * pl.col('day_of_year').cos() / 365).alias('day_of_year_cos').cast(pl.Float32),
        (2 * np.pi * pl.col('day_of_year') / 365).cos().alias('cos_day_of_year_rad').cast(pl.Float32),
    )
    return df


if config.get('TimeFE', False):
    print('**********add time features**********')
    dfpl = get_time_feature(dfpl)
    print('Now, the shape of data is:', dfpl.shape)
'''
    2. Add the weather features
'''


@timeit
def get_weather_feature(df: pl.DataFrame):
    # 角度改变量
    df = df.with_columns(
        pl.col('10mWindDirection').diff(1).over('UserID').alias('10mWindDirection_diff'),
        pl.col('100mWindDirection').diff(1).over('UserID').alias('100mWindDirection_diff'),
        (pl.col('RelativeHumidity') * pl.col('Temperature')).alias('RelativeHumidity*Temperature'),
        (pl.col('RelativeHumidity') * pl.col('Radiation')).alias('RelativeHumidity*Radiation'),
        (pl.col('Radiation') / (pl.col('Temperature') + 1e-6)).alias('Radiation/Temperature'),
        (pl.col('Radiation') / (pl.col('CloudAmount') + 1e-6)).alias('Radiation/CloudAmount'),
        (pl.col('AirPressure') / pl.col('RelativeHumidity')).alias('AbsoluteHumidity'),
        (pl.col('AirPressure') * pl.col('Temperature')).alias('AirPressure*Temperature')
    )

    # 计算风角变化绝对值
    df = df.with_columns(
        pl.col('10mWindDirection_diff').abs().alias('10mWindDirection_diff_abs'),
        pl.col('100mWindDirection_diff').abs().alias('100mWindDirection_diff_abs')
    )

    return df


if config.get('WeatherFE', False):
    print('**********add weather features**********')
    dfpl = get_weather_feature(dfpl)
    print('Now, the shape of data is:', dfpl.shape)
'''
    3. statistic features
'''


@timeit
def stat_feature(df: pl.DataFrame, columns: list):
    half_day_list = ['UserID', 'year', 'month', 'day', 'half_day']
    hour_list = ['UserID', 'year', 'month', 'day', 'hour']
    df = df.with_columns(
        *[
            pl.col(col).mean().over(half_day_list).alias(f'{col}_half_day_mean') for col in columns
        ],
        *[
            pl.col(col).std().over(half_day_list).alias(f'{col}_half_day_std') for col in columns
        ],
        *[
            pl.col(col).max().over(half_day_list).alias(f'{col}_half day_max') for col in columns
        ],
        *[
            pl.col(col).kurtosis().over(half_day_list).alias(f'{col}_half_day_kurtosis') for col in columns
        ],
        *[
            pl.col(col).skew().over(half_day_list).alias(f'{col}_half_day_skew') for col in columns
        ],
        *[
            pl.col(col).mean().over(hour_list).alias(f'{col}_hour_mean') for col in columns
        ],
        *[
            pl.col(col).std().over(hour_list).alias(f'{col}_hour_std') for col in columns
        ],
        *[
            pl.col(col).max().over(hour_list).alias(f'{col}_hour_max') for col in columns
        ],
        *[
            pl.col(col).kurtosis().over(hour_list).alias(f'{col}_hour_kurtosis') for col in columns
        ],
        *[
            pl.col(col).skew().over(hour_list).alias(f'{col}_hour_skew') for col in columns
        ]
    )
    return df


if config.get('StatFE', False):
    print('**********add statistic features**********')
    columns = ['AirPressure', 'RelativeHumidity', 'CloudAmount', 'Temperature', 'Radiation'
        , 'RelativeHumidity*Temperature', 'RelativeHumidity*Radiation', 'Radiation/Temperature',
               'Radiation/CloudAmount', 'AirPressure*Temperature']
    dfpl = stat_feature(dfpl, columns=columns)
    print('Now, the shape of data is:', dfpl.shape)
'''
    4. lag features (shift feature)
'''


def shift_expr(col: str, pre_days=0, pre_hours=0, min_cnt=0) -> pl.Expr:
    # min_cnt = 15min * cnt
    shift_step = min_cnt + pre_hours * 4 + pre_days * 96
    new_name = str(pre_days) + 'days_' + str(pre_hours) + 'hours_' + str(min_cnt) + 'min_cnt_' + col
    return pl.col(col).shift(shift_step).over('UserID').alias(new_name)


@timeit
def get_shift_feature(df: pl.DataFrame, columns):
    # constructs the expr list
    expr_list = []
    for col in columns:
        for num in [-7, -3, -1, 0, 1, 3, 7]:
            expr_list.append(shift_expr(col, min_cnt=num))

        for day in [-1, 1]:
            for num in [-5, -2, 0, 2, 5]:
                expr_list.append(shift_expr(col, pre_days=day, min_cnt=num))
    df = df.with_columns(
        *expr_list
    )
    return df


if config.get('ShiftFE', False):
    print('**********add shift features**********')
    columns = ['AirPressure', 'RelativeHumidity', 'CloudAmount', 'Temperature', 'Radiation']
    dfpl = get_shift_feature(dfpl, columns)
    print('Now, the shape of data is:', dfpl.shape)

'''
    5. diff feature
'''


def get_diff_expr(col: str) -> pl.expr:
    return [
        pl.col(col).diff(1).over('UserID').alias(f'backward_{col}_diff1_order1'),
        pl.col(col).diff(-1).over('UserID').alias(f'forward_{col}_diff1_order1'),
        pl.col(col).diff(2).over('UserID').alias(f'backward_{col}_diff2_order1'),
        pl.col(col).diff(-2).over('UserID').alias(f'forward_{col}_diff2_order1'),
        pl.col(col).diff(1).diff(1).over('UserID').alias(f'backward_{col}_diff1_order2'),
        pl.col(col).diff(-1).diff(-1).over('UserID').alias(f'forward{col}_diff1_order2'),
    ]


@timeit
def get_diff_feature(df: pl.DataFrame, columns: list):
    expr_list = []
    for col in columns:
        expr_list.extend(get_diff_expr(col))
    df = df.with_columns(
        *expr_list
    )
    return df


if config.get('DiffFE', False):
    print('**********add diff features**********')
    columns = ['AirPressure', 'RelativeHumidity', 'CloudAmount', 'Temperature', 'Radiation']
    dfpl = get_diff_feature(dfpl, columns)
    print('Now, the shape of data is:', dfpl.shape)

'''
    6. rolling feature
'''


def get_rolling_expr(columns: list, period='45m', forward=False):
    expr_list = []
    direction = 'forward' if forward else 'backward'
    for col in columns:
        expr_list.extend([
            pl.col(col).mean().alias(f'{col}_mean_rolling_{period}' + direction),
            pl.col(col).ewm_mean(alpha=0.9).apply(lambda x:x[-1]).alias(f'{col}_ewm_0.9_mean_rolling_{period}' +
                                                                     direction),
            pl.col(col).ewm_mean(alpha=0.8).apply(lambda x:x[-1]).alias(f'{col}_ewm_0.8_mean_rolling_{period}' +
                                                                     direction),
        ]
        )
    return expr_list

@timeit
def get_rolling_feature(df: pl.DataFrame, columns: list):
    df = df.with_columns(
        # df.rolling('ts', period='45m', group_by='UserID').agg(
        #     get_rolling_expr(columns, period='45m', forward=False)
        # ),
        df.rolling('ts', period='105m', group_by='UserID').agg(
            get_rolling_expr(columns, period='105m', forward=False)
        ),
        # df.rolling('ts', period='225m', group_by='UserID').agg(
        #     get_rolling_expr(columns, period='225m', forward=False)
        # ),
        # df.rolling('ts', period='450m', group_by='UserID').agg(
        #     get_rolling_expr(columns, period='450m', forward=False)
        # ),
        # df.rolling('ts', period='600m', group_by='UserID').agg(
        #     get_rolling_expr(columns, period='600m', forward=False)
        # ),
    )
    return df
if config.get('RollingFE',False):
    print('**********add rolling features**********')
    columns = ['AirPressure', 'RelativeHumidity', 'CloudAmount', 'Temperature', 'Radiation']
    dfpl = get_rolling_feature(dfpl, columns)
    print('Now, the shape of data is:', dfpl.shape)

'''
    save the data
'''
print('Saving the data')
# if os.path.exists(config['output']):
#     os.makedirs(config['output'])
dfpl.to_pandas().to_csv(config['output'], index=False)
print('Finish!')
