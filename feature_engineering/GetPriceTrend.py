'''
Get the average $/sqft of houses sold right before each house is listed.
'''

import pandas as pd
import numpy as np
import sys
sys.path.append('~/PycharmProjects/Boston_housing/')

def outlier(x):
    return 1.5*(np.quantile(x,0.75)-np.quantile(x,0.25)) + np.quantile(x,0.75)

property_type = 'condo'
PROPERTY_TYPE = 'Condo/Co-op'

df = pd.read_csv('../data/raw/' + 'redfin_2020-01-19-08-17-26.csv')
df = df[df['PROPERTY TYPE'] == PROPERTY_TYPE]
df.pop('PRICE')
df.pop('SOLD DATE')
# only transaction file contain list date
df_transaction = pd.read_csv('../data/processed/' + 'Boston_%s_transaction.csv'%property_type,index_col=0)
df = df.join(df_transaction,how='inner')

df = df[['PROPERTY TYPE','LIST DATE','SOLD DATE','SOLD PRICE','$/SQUARE FEET']]
df['LIST DATE'] = pd.to_datetime(df['LIST DATE'])
df['SOLD DATE'] = pd.to_datetime(df['SOLD DATE'])
df = df.sort_values('LIST DATE')
df = df.dropna()

#price_outlier = outlier(df['SOLD PRICE'])
#df = df[df['SOLD PRICE'] < price_outlier]

# $ per sqft of the last sold house
df_last_ppsqft = df[['SOLD DATE','$/SQUARE FEET']].groupby('SOLD DATE').median()
df_last_ppsqft.columns = ['EST $/SQUARE FEET']
# house listed today is using average $/sqft from the last sold house
df_last_ppsqft = df_last_ppsqft.shift(1)
# predict $/sqft sold on sold date from the list date
df_ppsqft = df[['LIST DATE','$/SQUARE FEET']].set_index('LIST DATE')

# $/sqft = price sold on sold date
# Est $/sqft = price of last house sold before the list date of the current house
# index = list date (date when estimation is made)
df_joined = df_ppsqft.join(df_last_ppsqft, how='left') # left join: always keep each house listed
df_joined = df_joined.fillna(method='ffill') # use the last average sold price
df_joined.index = df.index
df_joined.pop('$/SQUARE FEET')

df_joined.to_csv('../data/processed/' + 'Boston_%s_price_trend.csv'%property_type)








