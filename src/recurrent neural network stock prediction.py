#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.layers import LSTM


# In[2]:


def generateGraph(train, test, title, xlabel, ylabel, legend, fileTitle):
    ax, fig = plt.subplots(figsize=(10, 8))
    fig.plot(train)
    fig.plot(test)
    fig.set_xlabel(xlabel)
    fig.set_ylabel(ylabel)
    fig.set_title('{} - {}'.format(title, fileTitle))
    fig.legend(legend)
    ax.savefig('./graphs/{}.png'.format(fileTitle))
    


# In[3]:


def readData(path, split):
    df = pd.read_csv(path)
#     print(df.head())
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    df = df['Close']
    train = df.iloc[:-split]
    test = df.iloc[-split:]
    dates = df.index[-split+1:]
    return train, test, dates


# In[4]:


def scaleData(train, test):
    sc= MinMaxScaler()
    train_sc = sc.fit_transform(train.values.reshape(-1, 1))
    test_sc = sc.transform(test.values.reshape(-1, 1))
    x_train = train_sc[:-1]
    y_train = train_sc[1:]

    x_test = test_sc[:-1]
    y_test = test_sc[1:]
    
    return x_train, y_train, x_test, y_test, sc


# In[5]:


def createModel(x_train, y_train):
    x_train_t = x_train[:, None]
    
    K.clear_session()
    
    early_stop = EarlyStopping(monitor='loss', patience=1, verbose=1)
    model = Sequential()
    model.add(LSTM(6, input_shape=(1, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train_t, y_train, epochs=100, batch_size=16, verbose=1, callbacks=[early_stop])
    return model


# In[6]:


def saveSheet(y_pred, y_test, dates, sc, fileName):
    df = pd.DataFrame()
    y_pred = sc.inverse_transform(y_pred)
    y_test = sc.inverse_transform(y_test)
    df['Date'] = dates
    df['Predicted Close'] = y_pred.round(2)
    df['Actual Close'] = y_test.round(2)
    df['% error'] = ((y_pred - y_test) / y_test*100).round(2)
    print(df)
    df.to_csv('./results/{}.csv'.format(fileName))


# In[7]:


def analyze(path, split, fileName):
    train, test, dates = readData(path, split)
    generateGraph(train,
                  test,
                  'Training and Testing Data split',
                  'Dates',
                  'Closing value',
                  ['train', 'test'],
                  '{}_train_test_split'.format(fileName)
                 )
    x_train, y_train, x_test, y_test, sc = scaleData(train, test)
    model = createModel(x_train, y_train)
    x_test_t = x_test[:, None]
    y_pred = model.predict(x_test_t)
    generateGraph(y_test,
                  y_pred,
                  'Model results vs Actual value',
                  'Future days',
                  'Closing Value',
                  ['actual', 'predicted'],
                  '{}_results'.format(fileName)
                 )
    print(y_pred.shape, y_test.shape, dates.shape)
    saveSheet(y_pred, y_test, dates, sc, fileName)
    
    


# In[8]:


import os

for filename in os.listdir('./data'):
    analyze('./data/{}'.format(filename), 180, filename)

