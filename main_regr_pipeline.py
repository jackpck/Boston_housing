import sys
sys.path.append('./functions')

import pandas as pd
import numpy as np
import warnings

from model_functions import RegrSwitcher, load_data
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer,TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder,PowerTransformer,StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF,TruncatedSVD
from sklearn.model_selection import GridSearchCV,train_test_split,TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,classification_report,roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import pickle

warnings.filterwarnings('ignore')

# load preprocessed data
df = load_data()
print('df.shape: ',df.shape)

# pop out potential targets and unnecessary features.
price_per_sqft = df.pop('$/SQUARE FEET')
list_price = df.pop('LIST PRICE').values
days_on_mkt = df.pop('DAYS ON MKT').values
sold_price = df.pop('SOLD PRICE').values
premium = df.pop('PREMIUM')
premium_sign = np.sign(premium)

######################
# PREDICT SOLD PRICE #
######################

Y = sold_price
Y = np.log10(Y)
features = df.columns.tolist()
print('features: ',features)
X = df.values

numerical_features = ['SQUARE FEET','YEAR BUILT','EST $/SQUARE FEET'] # require BoxCox transformation
text_features = ['REMARKS'] # require vectorization
categorical_features = ['HAS LOT','PROPERTY TYPE'] # require one hot encoding

numerical_columns = [features.index(x) for x in numerical_features]
text_columns = [features.index(x) for x in text_features]
categorical_columns = [features.index(x) for x in categorical_features]

# Setting up the ML pipeline
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
                  ('Switcher',RegrSwitcher())])

#print(p_tot.get_params().keys())

params = [{'Switcher__estimator':[RandomForestRegressor()],
           'preprocess__p_text__TFIDF__ngram_range':[(1,1)],
           'preprocess__p_text__tSVD__n_components':[8,9,10],
           'Switcher__estimator__n_estimators':[30,40],
           'Switcher__estimator__max_depth':[8,9]
           },
          {'Switcher__estimator':[GradientBoostingRegressor()],
           'preprocess__p_text__TFIDF__ngram_range':[(1,1)],
           'preprocess__p_text__tSVD__n_components':[8,9,10],
           'Switcher__estimator__n_estimators':[30,40],
           'Switcher__estimator__max_depth':[8,9],
           'Switcher__estimator__learning_rate':np.logspace(-2,1,4)
           }]


tscv = TimeSeriesSplit(n_splits=5)
regr = GridSearchCV(p_tot,param_grid=params,cv=tscv,scoring='r2')

Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size=0.25,shuffle=False)
regr.fit(Xtrain,Ytrain)

print('best params: ',regr.best_params_)
print('best score: ',regr.best_score_)

with open('./pickled_models/RF_all_property_sold_price.pkl', 'wb') as f:
    pickle.dump(regr, f)

Ypred = regr.predict(Xtest)

print('PRICE PREDICTION')
print('r2 score: ',r2_score(Ytest,Ypred))
mre = np.mean(np.abs(10**Ytest - 10**Ypred)/10**Ytest)
print('MRE: ',mre)


