# Install dependencies
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import tensorflow as tf # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Reading in data from CSV
df = pd.read_csv('./data/WIKI_AAL.csv', index_col='Date')
# Splitting data into training and testing with 0.3 split
train, test = train_test_split(df, test_size=0.3)


# Normalizing data

# Flatenning the predicted close column into a numpy array
train_close = train['Adj. Close'].values.reshape(-1, 1)
print(train_close)

# Creating a scaler to smoothen out the data
scaler = MinMaxScaler()

# Batches of 500
smoothing_window_size = 500
di = 0

# Normalizing each window
while di < len(train_close):
    # print(di, len(train_close[di:di+smoothing_window_size]))
    scaler.fit(train_close[di:di+smoothing_window_size])
    train_close[di:di+smoothing_window_size] = scaler.transform(train_close[di:di+smoothing_window_size])
    di += smoothing_window_size

# Normalize the extra values
scaler.fit(train_close[di-smoothing_window_size:])
train_close[di-smoothing_window_size:] = scaler.transform(train_close[di-smoothing_window_size:])
train['Adj. Close'] = train_close
test['Adj. Close'] = scaler.transform(test['Adj. Close'].values.reshape(-1, 1))


# Now perform exponential moving average smoothing
# So the data will have a smoother curve than the original ragged data
EMA = 0.0
gamma = 0.1
for ti in range(len(train)):
  EMA = gamma*train['Adj. Close'][ti] + (1-gamma)*EMA
  train['Adj. Close'][ti] = EMA

# Used for visualization and test purposes
all_data = np.concatenate([train,test],axis=0)

forecast_out = 30
# Plotting data
# actual_data = df[['Adj. Close']]
# actual_data.plot(figsize=(10, 4))
# plt.show()
