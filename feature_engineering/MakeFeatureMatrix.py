import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')


def upper_outlier(x):
    return  np.quantile(x,0.75) + 1.5*(np.quantile(x,0.75)-np.quantile(x,0.25))

def lower_outlier(x):
    return  np.quantile(x,0.25) - 1.5*(np.quantile(x,0.75)-np.quantile(x,0.25))

property_type = 'condo'

df_topic = pd.read_csv('../data/processed/' + 'Boston_%s_topic_scores.csv'%property_type,index_col=0)
df_prop = pd.read_csv('../data/raw/' + 'redfin_2020-01-19-08-17-26.csv')
df_history = pd.read_csv('../data/processed/' + 'Boston_%s_transaction.csv'%property_type,index_col=0)
df_poi = pd.read_csv('../data/processed/' + 'Boston_%s_poi.csv'%property_type,index_col=0)
df_trend = pd.read_csv('../data/processed/' + 'Boston_%s_price_trend.csv'%property_type,index_col=0)
#df_lstm = pd.read_csv('../data/processed/' + 'Boston_%s_lstm.csv'%property_type,index_col=0)

df_prop = df_prop[['YEAR BUILT','SQUARE FEET','$/SQUARE FEET','BEDS','BATHS','LOT SIZE','HOA/MONTH']]

df_prop['LOT SIZE'] = df_prop['LOT SIZE'].fillna(0) # zero lot size if there is none
df_prop['HOA/MONTH'] = df_prop['HOA/MONTH'].fillna(0)

df = df_prop.join(df_topic, how = 'inner')
df = df.join(df_history, how = 'inner')
df = df.join(df_poi, how = 'inner')
df = df.join(df_trend, how = 'inner')
#df = df.join(df_lstm, how = 'inner',rsuffix='_LSTM')

print(df.columns)


df['SOLD DATE'] = pd.to_datetime(df['SOLD DATE'])
df['LIST DATE'] = pd.to_datetime(df['LIST DATE'])
df['DAYS ON MKT'] = (df['SOLD DATE']-df['LIST DATE']).apply(lambda x: x.days)

df = df[(df['DAYS ON MKT'] < upper_outlier(df['DAYS ON MKT']))&
        (df['SOLD PRICE'] < upper_outlier(df['SOLD PRICE'])) &
        (df['LIST PRICE'] < upper_outlier(df['LIST PRICE'])) &
        (df['DAYS ON MKT'] > lower_outlier(df['DAYS ON MKT'])) &
        (df['SOLD PRICE'] > lower_outlier(df['SOLD PRICE'])) &
        (df['LIST PRICE'] > lower_outlier(df['LIST PRICE']))]


df = df.dropna()

N = len(df)
train_size = int(0.7*N)

n_topic = 10

X_topic = df['REMARKS'].apply(lambda x: list(map(lambda x: float(x), x.split(' '))))
df_topic_vectorized = pd.DataFrame({'REMARKS_%i'%(i+1):[x[i] for x in X_topic] for i in range(n_topic)},
                                   index=X_topic.index)

df['EST $ TREND'] = df['EST $/SQUARE FEET']*df['SQUARE FEET']
#df['EST $ LSTM'] = df['EST $/SQUARE FEET_LSTM']*df['SQUARE FEET']

df_all = pd.DataFrame()
df_all = pd.concat([df_all,df[['convenience','supermarket','park','school','station','stop_position']]],axis=1)
df_all = pd.concat([df_all,df[['LIST PRICE','SQUARE FEET','YEAR BUILT','BEDS','BATHS','LOT SIZE','HOA/MONTH']]],axis=1)
df_all = pd.concat([df_all,df['EST $ TREND']],axis=1)
#df_all = pd.concat([df_all,df['EST $ LSTM']],axis=1)
df_all = pd.concat([df_all,df_topic_vectorized],axis=1)
df_all = pd.concat([df_all,df['SOLD PRICE']],axis=1)
df_all = pd.concat([df_all,df['SOLD DATE']],axis=1)

df_all = df_all.sort_values('SOLD DATE')

df_all.to_csv('../data/processed/' + 'Boston_%s_feature_matrix.csv'%property_type)
