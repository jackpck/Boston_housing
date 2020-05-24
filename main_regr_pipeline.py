import sys
sys.path.append('./functions')
import pandas as pd
import numpy as np
import warnings
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer,TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder,PowerTransformer,StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF,TruncatedSVD
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.base import BaseEstimator
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

property_type = 'single_family_residential'
df = pd.read_csv('./data/raw_joined/' + 'Boston_%s_joined_dataframe.csv'%property_type)
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
X = df.values

numerical_features = ['SQUARE FEET','YEAR BUILT'] # require BoxCox transformation
text_features = ['REMARKS'] # require vectorization
categorical_features = ['HAS LOT'] # require one hot encoding

numerical_columns = [features.index(x) for x in numerical_features]
text_columns = [features.index(x) for x in text_features]
categorical_columns = [features.index(x) for x in categorical_features]

class RegrSwitcher(BaseEstimator):
    def __init__(self,estimator=RandomForestRegressor()):
        self.estimator = estimator

    def fit(self,x,y=None,**kwargs):
        self.estimator.fit(x,y)
        return self

    def predict(self,x,y=None):
        return self.estimator.predict(x)

    def score(self,x,y):
        return self.estimator.score(x,y)



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


regr = GridSearchCV(p_tot,param_grid=params,cv=5,scoring='r2')

Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size=0.25,shuffle=False)
regr.fit(Xtrain,Ytrain)

print(regr.best_params_)
print(regr.best_score_)

Ypred = regr.predict(Xtest)


print('PRICE PREDICTION')
print('r2 score: ',r2_score(Ytest,Ypred))
print('MAE: ',mean_absolute_error(Ytest,Ypred))

#plt.scatter(Ytest,Ypred)
#plt.plot(Ytest,Ytest,'k--')
#plt.show()
