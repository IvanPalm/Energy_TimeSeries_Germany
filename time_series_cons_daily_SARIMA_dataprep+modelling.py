import pandas as pd

opsd = pd.read_csv('time_series_DE_daily.csv', index_col='date', parse_dates=True)
opsd.columns = opsd.columns.str.lower()

opsd.head(3)

# Subset electric consumption time series and resample by month
cons = opsd.loc[:, 'consumption'].dropna().resample('M').sum()
cons.describe()
cons.shape
cons.index[0]
cons.index[-1]

# Decompose time series
from statsmodels.tsa.seasonal import seasonal_decompose

cons_multi_dec = seasonal_decompose(cons, model='multiplicative', freq=12)
cons_addit_dec = seasonal_decompose(cons, model='additive', freq=12)

import matplotlib.pyplot as plt

# plt.rcParams.keys()
plt.rcParams.update({'figure.figsize':(15, 4)})
cons_multi_plot = cons_multi_dec.plot()
cons_addit_plot = cons_addit_dec.plot()
# residuals have same shape but higher range than in multiplicative

# Rolling windows - statistical stationarity
# cons.rolling(window=3, center=True).mean().head()

ws = 3
lw_th = 2
a = .3
lw = 2.2

ax = plt.plot(cons.index, cons, label='Raw data', color='r', linewidth=lw_th, alpha=a)
plt.plot(cons.rolling(window=ws, center=True).mean(), label='Rolling mean', linewidth=lw)
plt.plot(cons.rolling(window=ws, center=True).std(), label='Rolling std', linewidth=lw)
plt.title(f'Electricity consumption on a {ws} months rolling window', fontsize=16)
plt.xlabel('Year')
plt.ylabel('Electric consumption in Germany (GWh)')
plt.legend(fontsize=16, loc='best')

# Augmented Dickey-Fuller Test for stationarity
from statsmodels.tsa.stattools import adfuller

def ADF_Stationarity_Test(timeseries):
    ADF_test = adfuller(timeseries, autolag='AIC')
    pValue = ADF_test[1]

    if (pValue < .05):
        print('The time-series is stationary!\n')
    else:
        print('The time-series is NOT stationary!\n')

    # Coefficients
    dfResults = pd.Series(ADF_test[0:4], index=['ADF Test Statistic', 'P-Value', 'N-Lags Used', 'N-Observations Used'])
    #Critical values
    for key, value in ADF_test[4].items():
        dfResults['Critical value (%s)'%key] = value

    print('Augmented Dickey-Fuller Test results:')
    print(dfResults)

ADF_Stationarity_Test(timeseries=cons) # d=0

# Transform time series to achieve stationarity
# cons_diff = cons.diff().dropna() # d = 1

# ADF_Stationarity_Test(timeseries=cons_diff)

# Autocorrelation

pd.plotting.lag_plot(cons)

### Seasonal ARIMA # SARIMA(p, d, q)(P, D, Q)s

# Autocorrelation and partial autocorrelation function
k_cons = int(len(cons)/13)
k_cons < len(cons)/k_cons

# Correlograms
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

cons_acf_plot = plot_acf(cons, lags=k_cons) # MA[q] = 1
cons_pacf_plot = plot_pacf(cons, lags=k_cons) # AR[p] = 1

# Seasonal differencing
ax = plt.plot(cons, label='Original series', color='r')
plt.plot(cons.diff(12), label='Seasonal differencing', color='g')
plt.xlabel('Year')
plt.ylabel('Electric consumption Germany(GWh)')
plt.legend(loc='center left')
plt.tick_params(labelrotation=90)

k = 12 # our S
k_acf_plot = plot_acf(cons, lags=k) # AR[P] = 1; MA[Q] = 1

# Define training and test datasets
training_perc = .8
cons_split = int(len(cons) * training_perc)
cons_train, cons_test = cons[:cons_split], cons[cons_split:]

from statsmodels.tsa.statespace.sarimax import SARIMAX

# Build and fit the model
p = 1
d = 0
q = 1

P = 1
D = 1
Q = 1
S = 12

#help(SARIMAX)
cons_model = SARIMAX(cons_train, order=(p, d, q),
                     seasonal_order=(P, D, Q, S),
                     enforce_stationarity=False,
                     enforce_inveritbility=False)

cons_fit = cons_model.fit(disp=False)

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

cons_diagnostic = cons_fit.plot_diagnostics(figsize=(14, 9))

# Validation with static method
cons_predict = cons_fit.get_prediction(start=cons.index[cons_split], end=cons.index[-1], dynamic=False)
cons_pred_confint = cons_predict.conf_int()

y_predict = cons_predict.predicted_mean
y_observed = cons[cons_split:]

import numpy as np

rmse = np.sqrt(((y_predict - y_observed)**2).mean()); print('The Mean Squared Error is {}'.format(round(rmse, 2)))

ax = cons.plot(color='b', label='Original series', figsize=(20, 6))
cons_predict.predicted_mean.plot(ax=ax, color = 'r', label='Prediction with static method', alpha=.5)
ax.set_ylim(30000, 60000)
ax.fill_between(cons_pred_confint.index,
                cons_pred_confint.iloc[:, 0],
                cons_pred_confint.iloc[:, 1], color = 'k', alpha=.2)
ax.legend(loc='upper left', fontsize=16)

# Forecast
cons_forecast = cons_fit.get_forecast(steps='2021-12-31')
cons_fore_confint = cons_forecast.conf_int()

ax = cons.plot(color='b', label='Original series', figsize=(20, 6))
cons_forecast.predicted_mean.plot(ax=ax, color = 'r', label='Forecasted series')
ax.fill_between(cons_fore_confint.index,
                cons_fore_confint.iloc[:, 0],
                cons_fore_confint.iloc[:, 1], color = 'k', alpha=.2)
ax.legend(loc='upper left', fontsize=16)
