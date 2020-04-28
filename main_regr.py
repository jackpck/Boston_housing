import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')

property_type = 'single_family_residential'


df_X = pd.read_csv('./data/processed/' + 'Boston_%s_feature_matrix.csv'%property_type,index_col=0)
df_lstm = pd.read_csv('./data/processed/' + 'Boston_%s_lstm.csv'%property_type,index_col=0)
df_X = df_X.join(df_lstm,how='inner')
print(df_X.columns)

#pop_list = ['convenience','supermarket','park','school','station','stop_position','LIST PRICE']
pop_list = ['LIST PRICE','SOLD DATE']
for i in pop_list:
    df_X.pop(i)
#for i in range(10):
#    df_X.pop('REMARKS_%i'%(i+1))

Y = df_X.pop('SOLD PRICE').values
X = df_X.values
feature_names = df_X.columns
print(feature_names)

N = len(X)
print('number of data: ',N)
Nfeature = len(feature_names)
print('number of features: ',Nfeature)
train_size = int(0.6 * N)
val_size = int(0.6 * (N - train_size))
test_size = N - train_size - val_size

Xtrain = X[:train_size]
Xval = X[train_size:train_size+val_size]
Xtest = X[train_size+val_size:]

Ytrain = Y[:train_size]
Yval = Y[train_size:train_size+val_size]
Ytest = Y[train_size+val_size:]

print('Xtrain.shape: ',Xtrain.shape)
print('Xval.shape: ',Xval.shape)
print('Xtest.shape: ',Xtest.shape)
print('Ytrain.shape: ',Ytrain.shape)
print('Yval.shape: ',Yval.shape)
print('Ytest.shape: ',Ytest.shape)

params = {'max_depth':[6,7,8,9,10],'n_estimators':[10,20,30,40,50],
          'learning_rate':[0.01,0.1,1]}

regr_cv = GridSearchCV(GradientBoostingRegressor(),
                       params,
                       cv=5)

print('training')
regr_cv.fit(Xtrain,Ytrain)

best_param = regr_cv.best_params_

regr_score = GradientBoostingRegressor(max_depth=best_param['max_depth'],
                                   n_estimators=best_param['n_estimators'])

cv_score = cross_val_score(regr_cv, Xtrain, Ytrain, cv=5)
print('regr score: %.3f +/- %.5f'%(np.mean(cv_score),np.std(cv_score)))
print('best param: ',regr_cv.best_params_)

Yval_pred = regr_cv.predict(Xval)

print('mean relative error: ',np.mean(np.abs(Yval_pred-Yval)/Yval))
print('r2 score: ',r2_score(Yval,Yval_pred))
