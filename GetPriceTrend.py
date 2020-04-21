import pandas as pd
import numpy as np
import sys
import re
sys.path.append('~/PycharmProjects/Boston_housing/')

def outlier(x):
    return 1.5*(np.quantile(x,0.75)-np.quantile(x,0.25)) + np.quantile(x,0.75)

property_type = 'single_family_residential'
PROPERTY_TYPE = 'Single Family Residential'

df = pd.read_csv('./data/raw/' + 'redfin_2020-01-19-08-17-26.csv')
df = df[['PROPERTY TYPE','SOLD DATE','PRICE','$/SQUARE FEET']]
df = df[df['PROPERTY TYPE'] == PROPERTY_TYPE]
df = df.dropna()

price_outlier = outlier(df['PRICE'])
df = df[df['PRICE'] < price_outlier]
df['SOLD DATETIME'] = pd.to_datetime(df['SOLD DATE'])
df = df.sort_values('SOLD DATETIME')

# predict $/sqft on the next house by the average $/sqft of all properties sold in the last record date
df_average_price = df[['SOLD DATE','$/SQUARE FEET']].groupby('SOLD DATE').mean()
df_average_price['SOLD DATETIME'] = pd.to_datetime(df_average_price.index)
df_average_price = df_average_price.sort_values('SOLD DATETIME')

df_est_price = df_average_price.shift(1)

dict_date_price = {df_est_price.index[i]:
              df_est_price['$/SQUARE FEET'].iloc[i] for i in range(len(df_est_price))}

df_est_price_per_sqft = df['SOLD DATE'].apply(lambda x: dict_date_price[x])

df_est_price_per_sqft.name = 'EST $/SQUARE FEET'
df_est_price_per_sqft.to_frame().to_csv('./data/processed/' + 'Boston_%s_price_trend.csv'%property_type)








