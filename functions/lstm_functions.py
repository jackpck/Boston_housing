from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np

import matplotlib.pyplot as plt

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        dataX.append(dataset[i:i+look_back, 0])
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def data_splitting(dataset,test_size=0.3):
    N = len(dataset)
    train_size = int(N*(1-test_size))
    test_size = N - train_size

    train = dataset[:train_size]
    test = dataset[train_size:]

    return train,test

def data_reshaping(X,extend_axis='timestep'):
    if extend_axis == 'timestep':
        return X.reshape(X.shape[0],X.shape[1],1)
    elif extend_axis == 'feature':
        return X.reshape(X.shape[0],1,X.shape[1])
    else:
        print('extend_axis must either be timestep or feature')

def data_scaler(X):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(X)
    return scaler


def build_lstm(out_dim,timestep,nfeature):
    model = Sequential()
    model.add(LSTM(out_dim, input_shape=(timestep,nfeature)))  # dimensionality of the output space,
                                                         # input_shape=(number of time step, feature size)
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model


def plot_prediction(Ytrain,Ytest,Ytrain_pred,Ytest_pred,scaler,look_back,n):
    Ytrain = scaler.inverse_transform(Ytrain.reshape(-1,1))
    Ytrain_pred = scaler.inverse_transform(Ytrain_pred.reshape(-1,1))
    Ytest = scaler.inverse_transform(Ytest.reshape(-1,1))
    Ytest_pred = scaler.inverse_transform(Ytest_pred.reshape(-1,1))

    print('plot_prediction')
    print('Ytrain.shape:' ,Ytrain.shape)
    print('Ytest.shape: ',Ytest.shape)
    print('Ytrain_pred.shape:' ,Ytrain_pred.shape)
    print('Ytest_pred.shape: ',Ytest_pred.shape)


    Y = np.zeros((n ,1))
    Y[:,:] = np.nan
    Y[look_back:len(Ytrain_pred) + look_back,:] = Ytrain
    Y[len(Ytrain_pred) + (look_back*2) + 1:n - 1, :] = Ytest

    # shift train predictions for plotting
    trainPredictPlot = np.zeros((n,1))
    trainPredictPlot[:,:] = np.nan
    trainPredictPlot[look_back:len(Ytrain) + look_back,:] = Ytrain_pred

    # shift test predictions for plotting
    testPredictPlot = np.zeros((n,1))
    testPredictPlot[:,:] = np.nan
    testPredictPlot[len(Ytrain) + (look_back*2) + 1:n - 1, :] = Ytest_pred

    # plot baseline and predictions
    fig = plt.figure(figsize=(10, 10))
    plt.plot(Y)
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()


'''
if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import mean_squared_error

    sns.set()

    import sys

    sys.path.append('~/PycharmProjects/Boston_housing/')

    df = pd.read_csv('../data/processed/' + 'Boston_condo_transaction.csv', index_col=0)
    df_master = pd.read_csv('../data/raw/' + 'redfin_2020-01-19-08-17-26.csv')
    df_master.pop('SOLD DATE')

    df = df.join(df_master, how='inner')


    def upper_outlier(x):
        return np.quantile(x, 0.75) + 1.5 * (np.quantile(x, 0.75) - np.quantile(x, 0.25))


    def lower_outlier(x):
        return np.quantile(x, 0.25) - 1.5 * (np.quantile(x, 0.75) - np.quantile(x, 0.25))


    df['SOLD DATE'] = pd.to_datetime(df['SOLD DATE'])
    df['LIST DATE'] = pd.to_datetime(df['LIST DATE'])
    df = df[(df['SOLD PRICE'] < upper_outlier(df['SOLD PRICE'])) &
            (df['SOLD PRICE'] > lower_outlier(df['SOLD PRICE']))]


    df['DAYS ON MKT'] = (df['SOLD DATE'] - df['LIST DATE']).apply(lambda x: x.days)
    df = df[df['DAYS ON MKT'] > 0]  # price history scraped after the spreadsheet
    df = df[np.abs((df['SOLD PRICE'] - df['LIST PRICE']) / df['LIST PRICE']) < 1]  # typo in listing/selling price

    df = df.sort_values('SOLD DATE')


    df_price_per_sqft = df[['SOLD DATE', '$/SQUARE FEET']].groupby('SOLD DATE').mean()
    df_rolling_price_per_sqft = df_price_per_sqft.rolling(5).mean()
    df_rolling_price_per_sqft = df_rolling_price_per_sqft.dropna()
    n_data = len(df_rolling_price_per_sqft)

    rolling_price_per_sqft = df_rolling_price_per_sqft.values.reshape(-1,1)
    train,test = data_splitting(rolling_price_per_sqft)

    scaler = data_scaler(train)
    train = scaler.transform(train)
    test = scaler.transform(test)

    look_back = 5
    Xtrain,Ytrain = create_dataset(train,look_back=look_back)
    Xtest,Ytest = create_dataset(test,look_back=look_back)

    Xtrain = data_reshaping(Xtrain,extend_axis='timestep')
    Xtest = data_reshaping(Xtest,extend_axis='timestep')

    print('Xtrain.shape: ',Xtrain.shape)
    print('Xtest.shape:' ,Xtest.shape)
    print('Ytrain.shape:' ,Ytrain.shape)
    print('Ytest.shape: ',Ytest.shape)


    lstm = build_lstm(10,look_back,1)
    lstm.fit(Xtrain, Ytrain, epochs=50, batch_size=1, verbose=2)

    Ytrain_pred = lstm.predict(Xtrain)
    Ytest_pred = lstm.predict(Xtest)


    print('MSE: ',mean_squared_error(Ytest,Ytest_pred.reshape(Ytest_pred.shape[0],)))
    plot_prediction(Ytrain,
                    Ytest,
                    Ytrain_pred,
                    Ytest_pred,
                    scaler,look_back,
                    n_data)
'''


