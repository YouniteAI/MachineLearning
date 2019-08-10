'''

Regression Intro

'''

import pandas as pd
import quandl
import math
import datetime
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

pd.options.display.max_columns = 20

df = quandl.get("EOD/MCD", authtoken="1QcUxfH8pzXmp1zZKGR3", start_date="2010-01-01", end_date="2018-01-01")

df = df[['Adj_Open', 'Adj_High', 'Adj_Low', 'Adj_Close', 'Adj_Volume']]

df['HL_PCT'] = (df['Adj_High'] - df['Adj_Close']) / df['Adj_Close'] * 100
df['PCT_change'] = (df['Adj_Close'] - df['Adj_Open']) / df['Adj_Open'] * 100

df = df[['Adj_Close', 'HL_PCT', 'PCT_change', 'Adj_Volume']]

forecast_col = 'Adj_Close'  # label
df.fillna(-99999, inplace=True)   # replace the NaN data with something and treat as an outlier

forecast_out = int(math.ceil(0.1*len(df)))  # we're gonna forecast_out using 1% of the dataframe

df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))  # features is everything besides label column
X = preprocessing.scale(X)  # preprocess all the features
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])  # label is the label column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
'''

model = LinearRegression(n_jobs=-1)  # whenever you can use this you should
model.fit(X_train, y_train)
with open('linearRegression.pickle', 'wb') as f:
    pickle.dump(model, f)
'''

pickle_in = open('linearRegression.pickle', 'rb')
model = pickle.load(pickle_in)

accuracy = model.score(X_test, y_test)

# print(accuracy)

'''
Forecasting and predicting
'''

forecast_set = model.predict(X_lately)
print(forecast_set, accuracy, forecast_out)     # next 11 days of forecasted values
df['Forecast'] = np.nan


'''
Plotting
'''

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj_Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()