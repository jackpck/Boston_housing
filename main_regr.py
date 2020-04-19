import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')

def outlier(x):
    return 1.5*(np.quantile(x,0.75)-np.quantile(x,0.25)) + np.quantile(x,0.75)


property_type = 'condo'
predictor = 'DAYS ON MKT'

df_topic = pd.read_csv('./data/processed/' + 'Boston_%s_topic_scores.csv'%property_type,index_col=0)
df_prop = pd.read_csv('./data/raw/' + 'redfin_2020-01-19-08-17-26.csv')
df_history = pd.read_csv('./data/processed/' + 'Boston_%s_transaction.csv'%property_type,index_col=0)
df_poi = pd.read_csv('./data/processed/' + 'Boston_%s_poi.csv'%property_type,index_col=0)

df_prop = df_prop[['YEAR BUILT','SQUARE FEET','$/SQUARE FEET','BEDS','BATHS','LOT SIZE']]

df_prop['LOT SIZE'] = df_prop['LOT SIZE'].fillna(0)
print(df_prop['LOT SIZE'])

df = df_prop.join(df_topic, how = 'inner')
df = df.join(df_history, how = 'inner')
df = df.join(df_poi, how = 'inner')

print(len(df))
print(df.columns)

df['SOLD DATE'] = pd.to_datetime(df['SOLD DATE'])
df['LIST DATE'] = pd.to_datetime(df['LIST DATE'])
df['DAYS ON MKT'] = (df['SOLD DATE']-df['LIST DATE']).apply(lambda x: x.days)

df_outlier = df[['DAYS ON MKT','SOLD PRICE']].apply(lambda x: outlier(x),axis=0)

df = df[(df['DAYS ON MKT'] < df_outlier['DAYS ON MKT']) &
        (df['SOLD PRICE'] < df_outlier['SOLD PRICE'])]

df = df.dropna()

N = len(df)
train_size = int(0.7*N)

X_temp = df['REMARKS'].apply(lambda x: list(map(lambda x: float(x), x.split(' '))))
X = []

for i in range(len(X_temp)):
    X.append(X_temp.iloc[i])

X = np.array(X)
X = np.hstack([X,df[['convenience','supermarket','park','school','station','stop_position']].values])
X = np.hstack([X,df[['SQUARE FEET','YEAR BUILT','BEDS','BATHS','LOT SIZE']].values])

Xtrain = X[:train_size]
Xtest = X[train_size:]

Y = df[predictor].values
Ytrain = Y[:train_size]
Ytest = Y[train_size:]

print('Xtrain.shape: ',Xtrain.shape)
print('Xtest.shape: ',Xtest.shape)
print('Ytrain.shape: ',Ytrain.shape)
print('Ytest.shape: ',Ytest.shape)

params = {'max_depth':[2,3,4,5,6,7,8],'n_estimators':[10,20,30,40,50]}

regr_cv = GridSearchCV(RandomForestRegressor(),
                       params,
                       cv=5)

print('training')
regr_cv.fit(Xtrain,Ytrain)

best_param = regr_cv.best_params_

regr_score = RandomForestRegressor(max_depth=best_param['max_depth'],
                                   n_estimators=best_param['n_estimators'])

cv_score = cross_val_score(regr_cv, Xtrain, Ytrain, cv=5)
print('regr score: %.3f +/- %.5f'%(np.mean(cv_score),np.std(cv_score)))

Ypred = regr_cv.predict(Xtest)

print('mean relative error: ',np.mean(np.abs(Ypred-Ytest)/Ytest))
print('r2 score: ',r2_score(Ytest,Ypred))
