## This Python 3 environment comes with many helpful analytics libraries installed
## It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
## For example, here's several helpful packages to load
import os
import os.path as osp
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tqdm
import gc
import seaborn as sns
from sklearn.preprocessing import LabelEncoder



import datetime
import pytz
## Input data files are available in the read-only "../input/" directory
## For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
## import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
## You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
## You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

here = osp.dirname(osp.abspath(__file__))
data_dir = osp.join(here, '../data')

train_eval_dir = osp.join(data_dir, 'sales_train_evaluation.csv')
calendar_dir = osp.join(data_dir, 'calendar.csv')
price_dir = osp.join(data_dir, 'sell_prices.csv')

df_train_eval = pd.read_csv(train_eval_dir)
df_calendar = pd.read_csv(calendar_dir)
df_price = pd.read_csv(price_dir)

drop_days = ['d_%d'%i for i in range(1,1514)]
df_train_eval = df_train_eval.drop(drop_days, axis=1)
df_train_eval['mean'] = df_train_eval.apply(lambda x: np.mean(x.iloc[6:].to_numpy().astype(np.float)), axis=1)
df_train_eval['std'] = df_train_eval.apply(lambda x: np.std(x.iloc[6:].to_numpy().astype(np.float)), axis=1)
df_product = df_train_eval[['item_id', 'store_id', 'mean', 'std']]

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

## Making train data
def making_train_test_data(df_train_eval):
    print("processing train data")
    for i in range(28):
        df_train_eval['d_%d'%(1942+i)] = np.zeros((30490,)).astype(np.int)
    product_info_header = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'mean', 'std']
    train_header = product_info_header + ['d_%d'%i for i in range(1514,1970)]
    test_header = product_info_header + ['d_%d'%i for i in range(1870,1970)]
    
    df_train =  df_train_eval[train_header]
    df_test =  df_train_eval[test_header]
    df_train_after = pd.melt(df_train, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'mean', 'std'], var_name='days', value_name='demand')
    df_train_after['days'] = df_train_after['days'].map(lambda x: int(x[2:]))
    df_train_after = df_train_after.drop(['id'], axis=1)
    
    df_test_after = pd.melt(df_test, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'mean', 'std'], var_name='days', value_name='demand')
    df_test_after['days'] = df_test_after['days'].map(lambda x: int(x[2:]))
    df_test_after = df_test_after.drop(['id'], axis=1)
    
    df_train_after = reduce_mem_usage(df_train_after)
    df_test_after = reduce_mem_usage(df_test_after)
    gc.collect()
    return df_train_after, df_test_after

## Making calendar data
def making_calendar_data(df_calendar):
    df_calendar = reduce_mem_usage(df_calendar)
    gc.collect()
    print("processing calendar data")
    df_calendar['days'] = df_calendar['d'].map(lambda x: int(x[2:]))
    event_name =  {np.nan: 0,  'Halloween': 1, 'LentStart': 2, "Mother's day": 3, 'Cinco De Mayo': 4, 'EidAlAdha': 5,
                        'SuperBowl': 6, 'IndependenceDay': 7, 'StPatricksDay': 8, 'NBAFinalsEnd': 9, 'Easter': 10,
                        'MemorialDay': 11, 'ValentinesDay': 12, 'MartinLutherKingDay': 13, 'Christmas': 14,
                        'Purim End': 15, 'OrthodoxEaster': 16, 'Thanksgiving': 17, 'ColumbusDay': 18,
                        'VeteransDay': 19,
                        'NBAFinalsStart': 20, 'Pesach End': 21, 'LaborDay': 22, 'Chanukah End': 23,
                        'Eid al-Fitr': 24,
                        'LentWeek2': 25, 'NewYear': 26, 'PresidentsDay': 27, "Father's day": 28,
                        'OrthodoxChristmas': 29,
                        'Ramadan starts': 30}
    event_type = {np.nan: 0, 'Sporting': 1, 'Cultural': 2, 'National': 3, 'Religious': 4}
    df_calendar['event_name_1'] = df_calendar['event_name_1'].map(event_name)
    df_calendar['event_name_2'] = df_calendar['event_name_2'].map(event_name)
    df_calendar['event_type_1'] = df_calendar['event_type_1'].map(event_type)
    df_calendar['event_type_2'] = df_calendar['event_type_2'].map(event_type)
    df_calendar['year'] =  df_calendar['year'].map(lambda x: int(x - 2010))
    df_calendar = df_calendar.drop(['d', 'weekday', 'date'], axis=1)
    df_calendar = reduce_mem_usage(df_calendar)
    gc.collect()
    return df_calendar

## Making sell price data
def making_price_data(df_price):
    print("processing price data")
    df_price = reduce_mem_usage(df_price)
    gc.collect()
    return df_price

def concat_data(df_train_eval, df_calendar, df_price):
    df_train, df_test = making_train_test_data(df_train_eval)
    df_calendar = making_calendar_data(df_calendar)
    df_price = making_price_data(df_price)
    print("concat data")
    df_train = pd.merge(df_train, df_calendar, on='days', how='left')
    df_test = pd.merge(df_test, df_calendar, on='days', how='left')
    df_train = pd.merge(df_train, df_price, on=['wm_yr_wk', 'store_id', 'item_id'], how='left')
    df_test = pd.merge(df_test, df_price, on=['wm_yr_wk', 'store_id', 'item_id'], how='left')
    df_train = df_train.drop(['wm_yr_wk'], axis=1)
    df_test = df_test.drop(['wm_yr_wk'], axis=1)
    del df_calendar, df_price
    gc.collect()
    df_train = reduce_mem_usage(df_train)
    df_test = reduce_mem_usage(df_test)
    gc.collect()
    return df_train, df_test

def labeling_data(df_train_eval, df_calendar, df_price):
    df_train, df_test = concat_data(df_train_eval, df_calendar, df_price)
    print("labeling data")
    df_train['item_store'] = df_train['item_id'] + '_' + df_train['store_id']
    df_test['item_store'] = df_test['item_id'] + '_' + df_test['store_id']
    label_columns = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    for c in label_columns:
        le  = LabelEncoder()
        le.fit(df_train[c])
        df_train[c] = le.transform(df_train[c])
        df_test[c] = le.transform(df_test[c])
        if c != 'item_id':
            print(le.classes_)
    df_train = reduce_mem_usage(df_train)
    df_test = reduce_mem_usage(df_test)
    gc.collect()
    return df_train, df_test

start_time = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
df_train, df_test = labeling_data(df_train_eval, df_calendar, df_price)
df_train['available'] = df_train['sell_price'].map(lambda x: 0 if pd.isnull(x) else 1)
df_test['available'] = df_test['sell_price'].map(lambda x: 0 if pd.isnull(x) else 1)
df_train = df_train.fillna(np.inf)
df_test = df_test.fillna(np.inf)
df_train['demand_norm'] = df_train.apply(lambda x: 0 if x['std'] == 0 else ((x['demand'] - x['mean'])/x['std']), axis=1)
df_test['demand_norm'] = df_test.apply(lambda x: 0 if x['std'] == 0 else ((x['demand'] - x['mean'])/x['std']), axis=1)
df_train = reduce_mem_usage(df_train)
df_test = reduce_mem_usage(df_test)
df_product = reduce_mem_usage(df_product)
gc.collect()

df_train.to_csv(osp.join(here, 'train.csv'), index=False)
df_test.to_csv(osp.join(here, 'test.csv'), index=False)
df_product.to_csv(osp.join(here, 'product.csv'), index=False)
elapsed_time = (datetime.datetime.now(pytz.timezone('Asia/Tokyo')) - start_time).total_seconds()
print('Elapsed time for making train and test data', elapsed_time)