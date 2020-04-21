import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')

property_type = 'townhouse'


df_X = pd.read_csv('./data/processed/' + 'Boston_%s_feature_matrix.csv'%property_type,index_col=0)

Y = df_X.pop('SOLD PRICE').values
X = df_X.pop('LIST PRICE').values
feature_names = df_X.columns
print(feature_names)

N = len(X)
print('number of data: ',N)
Nfeature = len(feature_names)
print('number of features: ',Nfeature)
train_size = int(0.7*N)

Xtrain = X[:train_size]
Xtest = X[train_size:]

Ytrain = Y[:train_size]
Ytest = Y[train_size:]

print('Xtrain.shape: ',Xtrain.shape)
print('Xtest.shape: ',Xtest.shape)
print('Ytrain.shape: ',Ytrain.shape)
print('Ytest.shape: ',Ytest.shape)

Ypred = Xtest

print('mean relative error: ',np.mean(np.abs(Ypred-Ytest)/Ytest))
print('r2 score: ',r2_score(Ytest,Ypred))
