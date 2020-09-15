# Boston housing prediction

## Introduction
The increase availability of data in recent years have encouraged the use of multiple datasets and combining different modeling in data science projects. With domain specific insight and creativity, the use of unconventional datasets can also be valuable. In this project, I will predict the housing price in Boston using multiple datasets, ranging from numerics to text, and different machine learning models such as random forest and LSTM for regression, and non-negative matrix factorization for NLP topic modeling.

## Overview of the data used
Similar to the previous project `time2sell`, I obtained real estate data from `www.redfin.com`. The real estate data included the number of bedrooms, bathrooms, size, age, monthly HOA fee and the lot size of the property. It also contained the coordinate (latitude and longitude) of each house. The geospatial data is gathered from an online query system called Overpass turbo and save the data in `.json` file. The geospatial data contained the coordinates of all point of interests such as  convinience stores, supermarkets, parks, bus stops, train stations and schools in the greater Boston area. For each property, I scraped the broker's remark and the price history from `www.redfin.com` using selenium. A typical remark usually contained a more detail information about the property such as if the house is renovated recently, or if the property is close to some restaurents etc.

## Feature engineering
All the scripts for feature engineering can be found in `./feature_engineering`. For each house, I computed the number of point of interests within its 500m radius. `GetPOI.py` computed the feature array for a single point of interest (e.g. convenience store) and `MakePOI.py` combined the feature array of all point of interests into a feature matrix. For the brokers' remarks, I first preprocessed the text (removing stop words, lementizing, stemming, tokenizing). I then vectorize the words using tf-idf and find the latent dimension by using truncated-SVD. The number of components used in tSVD will be optimized using cross validation. Finally, I one-hot encoding all categorical features. Note that all feature engineering is done as part of the pipeline and also I use forward chaining cross validation to make sure I am not cheat by looking into the future.

![Image of the pipeline](https://github.com/jackpck/Boston_housing/blob/master/screenshots/pipeline.jpeg)

## Prediction
From my previous project `time2sell`, I have concluded that it was very difficult to predict the days on market (time between actual selling and listing) of the property. Therefore in this project I will just focused on predicting the selling price using extra and more interesting datasets. The forward chaining cross validation pipeline I built selected the best machine that yields the highest r2 score. In this case, the CV process chose Random Forest.  I used mean relative error (MRE) as the metric due to its interpretability (e.g. the prediction on average is different than the true value by 20%). The regression is done in `main_regr_pipeline.py`.

## Notebook
I encourage the reader to take a look at the notebooks in `./notebook` as they showed how my logic were formed and my understanding of the data through the exploratory data analysis (EDA), and how that understanding led to the choice of my model. I suggest reader to look at `eda.ipynb` for general EDA such as price trends in different neighborhoods, price distribution of different property types etc.; `lstm_eda.ipynb` for trends in daily average selling price and the use of LSTM; and `nlp.ipynb` for understanding what topics NMF found from the remarks.

## Web app
I have also created a web app `server.py` which allows user to estimate the house price based on the specifics of the house, the address and a brief remark of the house.
![Image of webapp frontpage](https://github.com/jackpck/Boston_housing/blob/master/screenshots/webapp_frontpage.png)
