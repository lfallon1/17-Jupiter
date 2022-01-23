# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 23:29:18 2022

@author: LF
"""


import yfinance as yf
# import requests
# from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import datetime as dt
import warnings
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
from pylab import rcParams
rcParams['figure.figsize'] = 10, 6
# import statsmodels.api as sm
from pylab import rcParams
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from pylab import rcParams
rcParams['figure.figsize'] = 10, 6
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math


tickers = ["AAL.L"]
#AAL.L - Anglo American
#BHP - BHP Group

financial_dir = {}

# =============================================================================

# Download stock price history data

start = dt.datetime.today()-dt.timedelta(3650)
end = dt.datetime.today()

for ticker in tickers:
    df = yf.download(ticker,start,end) 
    
#plot close price
plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Close Prices')
plt.plot(df['Close'])
plt.title(ticker+' - closing price')
plt.show()

# Lets us plot the scatterplot:
df_close = df['Close']
df_close.plot(style='k.')
plt.title(ticker+' - scatter plot of closing price')
plt.show()    


#Test for staionarity
def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    #Plot rolling statistics:
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show(block=False)
    
    print("Results of dickey fuller test")
    adft = adfuller(timeseries,autolag='AIC')
    # output for dft will give us without defining what the values are.
    #hence we manually write what values does it explains using a for loop
    output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
    for key,values in adft[4].items():
        output['critical value (%s)'%key] =  values
    print(output)
    
test_stationarity(df_close)
# Through the above graph, we can see the decreasing mean and standard deviation and 
# hence our series is not stationary.

# p value is above 0.05 so we cannot reject our NULL hypothesis, also test statistic is greater than our critical values
# so the data is non-stationary.

# In order to perform a time series analysis, we may need to separate seasonality and trend from our series. The resultant series will become stationary through this process.

# So let us separate Trend and Seasonality from the time series.

result = seasonal_decompose(df_close, model='multiplicative', freq = 30)
fig = plt.figure()  
fig = result.plot()  
fig.set_size_inches(16, 9)


# we start by taking a log of the series to reduce the magnitude of the values 
# and reduce the rising trend in the series. Then after getting the log of the series, 
# we find the rolling average of the series. A rolling average is calculated by taking 
# input for the past 12 months and giving a mean consumption value at every point further ahead in series.



rcParams['figure.figsize'] = 10, 6
df_log = np.log(df_close)
moving_avg = df_log.rolling(12).mean()
std_dev = df_log.rolling(12).std()
plt.legend(loc='best')
plt.title('Moving Average')
plt.plot(std_dev, color ="black", label = "Standard Deviation")
plt.plot(moving_avg, color="red", label = "Mean")
plt.legend()
plt.show()


# Now we are going to create an ARIMA model and will train it with the closing price of the stock on the train data. 
# So let us split the data into training and test set and visualize it.

#split data into train and training set
train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]
plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Closing Prices')
plt.plot(df_log, 'green', label='Train data')
plt.plot(test_data, 'blue', label='Test data')
plt.legend()



model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
                      test='adf',       # use adftest to find             optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

print(model_autoARIMA.summary())



# Before moving forward, let’s review the residual plots from auto ARIMA.
model_autoARIMA.plot_diagnostics(figsize=(15,8))
plt.show()

# Top left: The residual errors seem to fluctuate around a mean of zero and have a uniform variance.

# Top Right: The density plot suggest normal distribution with mean zero.

# Bottom left: All the dots should fall perfectly in line with the red line. Any significant deviations would imply the distribution is skewed.

# Bottom Right: The Correlogram, aka, ACF plot shows the residual errors are not autocorrelated. Any autocorrelation would imply that there is some pattern in the residual errors which are not explained in the model. So you will need to look for more X’s (predictors) to the model.



# ================================================
#model = sm.tsa.statespace.SARIMAX(train_data, order=(0, 1, 1))  

model = ARIMA(train_data, order=(2, 1, 1))   #ALL
# model = ARIMA(train_data, order=(2, 1, 0))   #BHP

fitted = model.fit(disp=-1)  
print(fitted.summary())


# Forecast
fc, se, conf = fitted.forecast(len(test_data), alpha=0.05)  # 95% confidence #replace 252 with len
fc_series = pd.Series(fc, index=test_data.index)
lower_series = pd.Series(conf[:, 0], index=test_data.index)
upper_series = pd.Series(conf[:, 1], index=test_data.index)
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train_data, label='training')
plt.plot(test_data, color = 'blue', label='Actual Stock Price')
plt.plot(fc_series, color = 'orange',label='Predicted Stock Price')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.10)
plt.title(ticker+' Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Actual Stock Price')
plt.legend(loc='upper left', fontsize=8)
plt.show()

# report performance
mse = mean_squared_error(test_data, fc)
print('MSE: '+str(mse))
mae = mean_absolute_error(test_data, fc)
print('MAE: '+str(mae))
rmse = math.sqrt(mean_squared_error(test_data, fc))
print('RMSE: '+str(rmse))
mape = np.mean(np.abs(fc - test_data)/np.abs(test_data))
print('MAPE: '+str(mape))


# =============================================================================
#Predicting price in two weeks time
# =============================================================================
#Create empty df for next two weeks
new_dates = pd.Series([])
new_dates.index = pd.DatetimeIndex(new_dates.index)
new_dates = new_dates.reindex(pd.date_range("2022-01-21", "2022-02-06"), fill_value="NaN")

new_test_data = pd.concat([test_data,new_dates])
new_train_data = train_data

model_autoARIMA = auto_arima(new_train_data, start_p=0, start_q=0,
                      test='adf',       # use adftest to find             optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

print(model_autoARIMA.summary())



# Before moving forward, let’s review the residual plots from auto ARIMA.
model_autoARIMA.plot_diagnostics(figsize=(15,8))
plt.show()

model = ARIMA(new_train_data, order=(0, 1, 1))  

fitted = model.fit(disp=-1)  
print(fitted.summary())
#Best model:  ARIMA(0,1,1)




# Forecast
fc, se, conf = fitted.forecast(len(new_test_data), alpha=0.05)  # 95% confidence #replace 252 with len
fc_series = pd.Series(fc, index=new_test_data.index)
lower_series = pd.Series(conf[:, 0], index=new_test_data.index)
upper_series = pd.Series(conf[:, 1], index=new_test_data.index)
plt.figure(figsize=(12,5), dpi=100)
plt.plot(new_train_data, label='training')
# plt.plot(new_test_data[-50:], color = 'blue', label='Actual Stock Price')
plt.plot(fc_series, color = 'orange',label='Predicted Stock Price')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.10)
plt.title(ticker+' Stock Price Prediction - Two Weeks Time')
plt.xlabel('Time')
plt.ylabel('Actual Stock Price')
plt.legend(loc='upper left', fontsize=8)
plt.show()


np.exp(fc_series[-1])
