import sys
sys.path.append('./functions')

import pandas as pd
import numpy as np
import warnings

from model_functions import RegrSwitcher
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer,TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder,PowerTransformer,StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF,TruncatedSVD
from sklearn.model_selection import GridSearchCV,train_test_split,TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import pickle

warnings.filterwarnings('ignore')

property_types = ['single_family_residential','condo','townhouse']

df = pd.DataFrame([])
for property_type in property_types:
    df_temp = pd.read_csv('./data/raw_joined/' + 'Boston_%s_joined_dataframe.csv'%property_type,index_col=0)
    df_temp['PROPERTY TYPE'] = property_type
    df = pd.concat([df,df_temp])

df = df.sort_values('SOLD DATE')
#df = df.sample(len(df))

df['DAYS ON MKT'] = df['DAYS ON MKT'].apply(lambda x: x if x > 0 else np.nan)
df['PREMIUM'] = (df['SOLD PRICE'] - df['LIST PRICE'])/df['LIST PRICE']
df['PREMIUM'] = df['PREMIUM'].apply(lambda x: x if np.abs(x) < 2 else np.nan)
df = df.dropna()

df.pop('$/SQUARE FEET')

df.pop('LIST DATE')
df.pop('SOLD DATE')
df.pop('EST $/SQUARE FEET')
df.pop('EST $ TREND')
df.pop('HOA/MONTH')
df.pop('PREMIUM')

df['HAS LOT'] = df['LOT SIZE'].apply(lambda x: 1 if x > 0 else 0)


list_price = df.pop('LIST PRICE').values
days_on_mkt = df.pop('DAYS ON MKT').values
sold_price = df.pop('SOLD PRICE').values


######################
# PREDICT SOLD PRICE #
######################

Y = sold_price
Y = np.log10(Y)
features = df.columns.tolist()
print(features)
X = df.values

numerical_features = ['SQUARE FEET','YEAR BUILT'] # require BoxCox transformation
text_features = ['REMARKS'] # require vectorization
categorical_features = ['HAS LOT','PROPERTY TYPE'] # require one hot encoding

numerical_columns = [features.index(x) for x in numerical_features]
text_columns = [features.index(x) for x in text_features]
categorical_columns = [features.index(x) for x in categorical_features]


p_num = Pipeline([('BoxCox',PowerTransformer(method='box-cox'))])
p_cat = Pipeline([('OneHot',OneHotEncoder(handle_unknown='ignore',sparse=False))])
p_text = Pipeline([('TFIDF',TfidfVectorizer()),
                   ('tSVD',TruncatedSVD())])

preprocess = ColumnTransformer(transformers=[
                               ('p_cat',p_cat,categorical_columns),
                               ('p_num',p_num,numerical_columns),
                               ('p_text',p_text,text_columns[0]) # [0] ensure the TfidfVectorizer acts on str,
                                                                 # not ndarray.
                            ])

p_tot = Pipeline([('preprocess',preprocess),
                  #('RF',RandomForestRegressor(n_jobs=-1))])
                  ('Switcher',RegrSwitcher())])

#print(p_tot.get_params().keys())

#params = {'preprocess__p_text__TFIDF__ngram_range':[(1,1)],
#          'preprocess__p_text__tSVD__n_components':[8,9,10],
#          'RF__n_estimators':[60,70],
#          'RF__max_depth':[8,9]
#          }

params = [{'Switcher__estimator':[RandomForestRegressor()],
           'preprocess__p_text__TFIDF__ngram_range':[(1,1)],
           'preprocess__p_text__tSVD__n_components':[8,9,10],
           'Switcher__estimator__n_estimators':[60,70],
           'Switcher__estimator__max_depth':[8,9]
           },
          {'Switcher__estimator':[GradientBoostingRegressor()],
           'preprocess__p_text__TFIDF__ngram_range':[(1,1)],
           'preprocess__p_text__tSVD__n_components':[8,9,10],
           'Switcher__estimator__n_estimators':[60,70],
           'Switcher__estimator__max_depth':[8,9],
           'Switcher__estimator__learning_rate':np.logspace(-2,1,4)
           }]


tscv = TimeSeriesSplit(n_splits=5)
regr = GridSearchCV(p_tot,param_grid=params,cv=tscv,scoring='r2')

Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size=0.25,shuffle=False)

print(Xtrain.shape)
print(Xtrain[0])
print(Xtest.shape)
print(Xtest[0])

regr.fit(Xtrain,Ytrain)

print(regr.best_params_)
print(regr.best_score_)


with open('./pickled_models/RF_all_property_sold_price.pkl', 'wb') as f:
    pickle.dump(regr, f)


Ypred = regr.predict(Xtest)

print('PRICE PREDICTION')
print('r2 score: ',r2_score(Ytest,Ypred))
mre = np.mean(np.abs(10**Ytest - 10**Ypred)/10**Ytest)
print('MRE: ',mre)

plt.scatter(Ytest,Ypred)
plt.plot(Ytest,Ytest,'k--')
plt.show()
