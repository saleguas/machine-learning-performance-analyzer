#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense

# Importing other files
import dataEditing
import externalData
from organizers import DatumHolder

ROLLING_WINDOW = 5
# Creates a LTSM model with the gixen X and Y variables. Returns a keras model


def createModelPrediction(x_train, y_train, x_pred):
    K.clear_session()
    x_train = x_train[:, None]
    early_stop = EarlyStopping(monitor='loss', patience=1, verbose=1)
    model = Sequential()
    model.add(LSTM(50, input_shape=(1, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=100, batch_size=16,
              verbose=1, callbacks=[early_stop])
    y_pred = model.predict(x_pred)
    return y_pred

# for s in range(1, 8):
#     x_pred['shift_{}'.format(s)] = x_pred.shift(s)
#     x_train['shift_{}'.format(s)] = x_pred.shift(s)
#     y_train['shift_{}'.format(s)] = x_pred.shift(s)

# Creates a report with the future predictions
def createPredictionProject(paths, future):
    allData = DatumHolder()

    for path in paths:
        fileName = os.path.basename(path)
        csvData = dataEditing.readData(path)
        x_train, y_train, x_pred, dates = dataEditing.getPredictData(
            csvData, future)

        scaler = MinMaxScaler()
        scaler.fit(csvData.values.reshape(-1, 1))
        x_train = scaler.transform(x_train.values.reshape(-1, 1))
        y_train = scaler.transform(y_train.values.reshape(-1, 1))
        x_pred = scaler.transform(x_pred.values.reshape(-1, 1))

        # x_train = pd.DataFrame(x_train)
        # y_train = pd.DataFrame(y_train)
        # x_pred = pd.DataFrame(x_pred)
        #
        # for s in range(1, ROLLING_WINDOW+1):
        #     x_pred['shift_{}'.format(s)] = x_pred.iloc[:, 0].shift(s)
        #     x_train['shift_{}'.format(s)] = x_train.iloc[:, 0].shift(s)
        #
        # x_pred = x_pred.dropna().values
        # x_train = x_train.dropna().values
        # y_train = y_train.dropna().iloc[:-ROLLING_WINDOW, ].values
        x_pred = x_pred[:, None]
        y_pred = createModelPrediction(x_train, y_train, x_pred)
        y_pred = scaler.inverse_transform(y_pred)

        y_pred = y_pred.flatten()

        allData.addRawSheet(fileName, y_pred, dates)

    externalData.createDataReport(allData, 'predict')

# Creates a report analyzing the model with a given split


def createAnalyzeProject(paths, split):
    allData = DatumHolder()

    for path in paths:
        fileName = os.path.basename(path)
        csvData = dataEditing.readData(path)
        scaleData, x_train, y_train, x_pred, y_true, dates = dataEditing.getAnalyzeData(
            csvData, split)

        scaler = MinMaxScaler()
        scaler.fit(scaleData.values.reshape(-1, 1))
        x_train = scaler.transform(x_train.values.reshape(-1, 1))
        y_train = scaler.transform(y_train.values.reshape(-1, 1))
        x_pred = scaler.transform(x_pred.values.reshape(-1, 1))

        x_pred = x_pred[:, None]
        y_pred = createModelPrediction(x_train, y_train, x_pred)
        y_pred = scaler.inverse_transform(y_pred)
        y_pred = y_pred.flatten()
        y_true = y_true.values

        allData.addAnalyzeSheet(fileName, y_pred, y_true, dates)

    externalData.createDataReport(allData, 'analyze')

# createPredictionProject(['../data/AMD.csv'], 30)
