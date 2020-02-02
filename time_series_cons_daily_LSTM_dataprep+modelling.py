import pandas as pd

opsd = pd.read_csv('time_series_DE_daily.csv', index_col='date', parse_dates=True)
opsd.columns = opsd.columns.str.lower()
#opsd.head(3)

cons = opsd.loc[:, 'consumption'].dropna().resample('M').sum()

# Test statio of the series
from statsmodels.tsa.stattools import adfuller

# def ADF_Stationarity_Test(timeseries):
#     ADF_test = adfuller(timeseries, autolag='AIC')
#     pValue = ADF_test[1]
#
#     if (pValue < .05):
#         print('The time-series is stationary!\n')
#     else:
#         print('The time-series is NOT stationary!\n')
#
#     # Coefficients
#     dfResults = pd.Series(ADF_test[0:4], index=['ADF Test Statistic', 'P-Value', 'N-Lags Used', 'N-Observations Used'])
#     #Critical values
#     for key, value in ADF_test[4].items():
#         dfResults['Critical value (%s)'%key] = value
#
#     print('Augmented Dickey-Fuller Test results:')
#     print(dfResults)
#
# ADF_Stationarity_Test(timeseries=cons)

# Prepare data for LSTM network
dataset = cons.values # numpy array

import numpy as np
from sklearn.preprocessing import MinMaxScaler

dataset.dtype
dataset = np.reshape(dataset, (-1, 1))
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Define training and test datasets
train_perc = .8
split_val = int(len(dataset) * train_perc)
train, test = dataset[:split_val], dataset[split_val:]

# train_size = int(len(dataset) * 0.80)
# test_size = len(dataset) - train_size
# train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

def create_samples(dataset, look_back):
    X, y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)

look_back = 18 # periodicity of data is 12
X_train, y_train = create_samples(train, look_back)
X_test, y_test = create_samples(test, look_back)

# reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# model architecture
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.callbacks import EarlyStopping

n_neurons = 60
model = Sequential()
model.add(LSTM(n_neurons, input_shape=(X_train.shape[1], X_train.shape[2]), activation='linear'))
#model.add(LSTM(100))
#model.add(Dropout(0.2))
#model.add(Dense(1))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')

n_epochs = 150
n_batches = 32
fitted = model.fit(X_train, y_train, epochs=n_epochs, batch_size=n_batches, validation_data=(X_test, y_test),
                    callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=0, shuffle=False)

model.summary()

# make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
# invert predictions
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])

print('Train Mean Absolute Error:', mean_absolute_error(y_train[0], train_predict[:,0]))
print('Train Root Mean Squared Error:',np.sqrt(mean_squared_error(y_train[0], train_predict[:,0])))
print('Test Mean Absolute Error:', mean_absolute_error(y_test[0], test_predict[:,0]))
print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(y_test[0], test_predict[:,0])))

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,4))
plt.plot(fitted.history['loss'], label='Train Loss')
plt.plot(fitted.history['val_loss'], label='Test Loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')

plt.figure(figsize=(12,6))
plt.plot(y_test[0], marker='.', label="Actual")
plt.plot(test_predict[:,0], marker='.', color='r', label="Prediction")
plt.tight_layout()
sns.despine(top=True)
plt.subplots_adjust(left=0.07)
plt.ylabel('Electricity consumption', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
