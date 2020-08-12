import sys
sys.path.append('../functions/')

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.metrics import r2_score
from nlp_functions import preprocess
import warnings

warnings.filterwarnings('ignore')


def upper_outlier(x):
    return  np.quantile(x,0.75) + 1.5*(np.quantile(x,0.75)-np.quantile(x,0.25))

def lower_outlier(x):
    return  np.quantile(x,0.25) - 1.5*(np.quantile(x,0.75)-np.quantile(x,0.25))

property_type = 'single_family_residential'

df_topic = pd.read_csv('../data/textdata/' + 'Boston_%s_remarks.csv'%property_type,index_col=0)
df_prop = pd.read_csv('../data/raw/' + 'redfin_2020-01-19-08-17-26.csv')
df_history = pd.read_csv('../data/processed/' + 'Boston_%s_transaction.csv'%property_type,index_col=0)
df_poi = pd.read_csv('../data/processed/' + 'Boston_%s_poi.csv'%property_type,index_col=0)
df_trend = pd.read_csv('../data/processed/' + 'Boston_%s_price_trend.csv'%property_type,index_col=0)

df_prop = df_prop[['YEAR BUILT','SQUARE FEET','$/SQUARE FEET','BEDS','BATHS','LOT SIZE','HOA/MONTH']]

df_prop['LOT SIZE'] = df_prop['LOT SIZE'].fillna(0) # zero lot size if there is none
df_prop['HOA/MONTH'] = df_prop['HOA/MONTH'].fillna(0)

df_topic = df_topic['REMARKS'].apply(lambda x: preprocess(x,pos_tag=['NNP','VBP']))

df = df_prop.join(df_topic, how = 'inner')
df = df.join(df_history, how = 'inner')
df = df.join(df_poi, how = 'inner')
df = df.join(df_trend, how = 'inner')

df['SOLD DATE'] = pd.to_datetime(df['SOLD DATE'])
df['LIST DATE'] = pd.to_datetime(df['LIST DATE'])
df['DAYS ON MKT'] = (df['SOLD DATE']-df['LIST DATE']).apply(lambda x: x.days)

print(df.isnull().sum())
df = df.dropna()
print(df.isnull().sum())

df = df.sort_values('LIST DATE')

df.to_csv('../data/raw_joined/' + 'Boston_%s_joined_dataframe.csv'%property_type)
