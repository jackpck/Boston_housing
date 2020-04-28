print(__name__)
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

import sys
import re

sys.path.append('../functions/')

from lstm_functions import data_splitting,data_scaler,data_reshaping,create_dataset,\
    build_lstm


def outlier(x):
    return 1.5 * (np.quantile(x, 0.75) - np.quantile(x, 0.25)) + np.quantile(x, 0.75)



property_type = 'townhouse'
PROPERTY_TYPE = 'Townhouse'

df = pd.read_csv('../data/raw/' + 'redfin_2020-01-19-08-17-26.csv')
df = df[['PROPERTY TYPE', 'SOLD DATE', 'PRICE', '$/SQUARE FEET']]
df = df[df['PROPERTY TYPE'] == PROPERTY_TYPE]
df = df.dropna()

price_outlier = outlier(df['PRICE'])
df = df[df['PRICE'] < price_outlier]
df['SOLD DATETIME'] = pd.to_datetime(df['SOLD DATE'])
df = df.sort_values('SOLD DATETIME')

# predict $/sqft on the next house by the average $/sqft of all properties sold in the last record date
df_average_price = df[['SOLD DATE', '$/SQUARE FEET']].groupby('SOLD DATE').mean()
df_average_price['SOLD DATETIME'] = pd.to_datetime(df_average_price.index)
df_average_price = df_average_price.sort_values('SOLD DATETIME')
df_average_price = df_average_price.set_index('SOLD DATETIME')
df_average_price.index = df_average_price.index.strftime('%Y-%m-%d')

df_average_price = df_average_price['$/SQUARE FEET']
print(df_average_price)

N = len(df_average_price)
train_size = int(0.6 * N)
val_size = int(0.6 * (N - train_size))
test_size = N - train_size - val_size


train, test = data_splitting(df_average_price, test_size=0.4)
train_index = train.index
test_index = test.index

train = train.values.reshape(-1,1)
test = test.values.reshape(-1,1)

scaler = data_scaler(train)
train = scaler.transform(train)
test = scaler.transform(test)

look_back = 5
Xtrain,Ytrain = create_dataset(train,look_back=look_back)
Xtest,Ytest = create_dataset(test,look_back=look_back)

Xtrain = data_reshaping(Xtrain,extend_axis='timestep')
Xtest = data_reshaping(Xtest,extend_axis='timestep')

lstm = build_lstm(10, look_back, 1)
lstm.fit(Xtrain, Ytrain, epochs=10, batch_size=1, verbose=2)

Ytrain_pred = lstm.predict(Xtrain)
Ytest_pred = lstm.predict(Xtest)

Ytrain_pred = scaler.inverse_transform(Ytrain_pred.reshape(-1, 1)).reshape(-1,)
Ytest_pred = scaler.inverse_transform(Ytest_pred.reshape(-1, 1)).reshape(-1,)
Ytrain = scaler.inverse_transform(Ytrain.reshape(-1, 1)).reshape(-1,)
Ytest = scaler.inverse_transform(Ytest.reshape(-1, 1)).reshape(-1,)

test_index = test_index[look_back+1:]

print('test_index.shape: ',test_index.shape)
print('Ytest_pred.shape: ',Ytest_pred.shape)

df_est_price = pd.DataFrame(Ytest_pred,columns=['EST $/SQUARE FEET'],index=test_index)
#print(df_val_est_price)

dict_date_price = {index:price for index,price in zip(df_est_price.index,df_est_price['EST $/SQUARE FEET'])}
df['str SOLD DATE'] = df['SOLD DATETIME'].apply(lambda x: x.strftime('%Y-%m-%d'))
df_est_price = df['str SOLD DATE'].apply(lambda x: dict_date_price[x] if x in dict_date_price else None)
df_est_price.name = 'EST $/SQUARE FEET'

df_est_price.to_frame().to_csv('../data/processed/Boston_%s_lstm.csv'%property_type)