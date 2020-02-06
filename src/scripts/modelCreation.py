#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

# Importing other files
import dataEditing
import externalData
from organizers import DatumHolder

# Creates a LTSM model with the gixen X and Y variables. Returns a keras model


def createModelPrediction(allData, split):

    sc = MinMaxScaler(feature_range=(0, 1))
    allData_scaled = sc.fit_transform(allData)
    training_set_scaled = allData_scaled[:-split]
    testing_set_scaled = allData_scaled[-split:]

    ##########################################################

    x_train = []
    y_train = []

    for i in range(60, len(training_set_scaled)):
        x_train.append(training_set_scaled[i - 60:i, 0])
        y_train.append(training_set_scaled[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    ##########################################################

    inputs = allData_scaled[len(allData_scaled) -
                            len(testing_set_scaled) - 60:]
    inputs = inputs.reshape(-1, 1)
    print(inputs)
    x_pred = []
    for i in range(60, len(inputs)):
        x_pred.append(inputs[i - 60:i, 0])

    x_pred = np.array(x_pred)
    x_pred = np.reshape(x_pred, (x_pred.shape[0], x_pred.shape[1],  1))
    y_true = sc.inverse_transform(testing_set_scaled)

    K.clear_session()
    early_stop = EarlyStopping(monitor='loss', patience=1, verbose=1)
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True,
                   input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=100, batch_size=32,
              verbose=1, callbacks=[early_stop])

    y_pred = model.predict(x_pred)
    y_pred = sc.inverse_transform(y_pred)



    return y_pred, y_true

# for s in range(1, 8):
#     x_pred['shift_{}'.format(s)] = x_pred.shift(s)
#     x_train['shift_{}'.format(s)] = x_pred.shift(s)
#     y_train['shift_{}'.format(s)] = x_pred.shift(s)

# # Creates a report with the future predictions
# def createPredictionProject(paths, future):
#     allData = DatumHolder()
#
#     for path in paths:
#         fileName = os.path.basename(path)
#         csvData = dataEditing.readData(path)
#         x_train, y_train, x_pred, dates = dataEditing.getPredictData(
#             csvData, future)
#
#         scaler = MinMaxScaler()
#         scaler.fit(csvData.values.reshape(-1, 1))
#         x_train = scaler.transform(x_train.values.reshape(-1, 1))
#         y_train = scaler.transform(y_train.values.reshape(-1, 1))
#         x_pred = scaler.transform(x_pred.values.reshape(-1, 1))
#         # x_train = pd.DataFrame(x_train)
#         # y_train = pd.DataFrame(y_train)
#         # x_pred = pd.DataFrame(x_pred)
#         #
#         # for s in range(1, ROLLING_WINDOW+1):
#         #     x_pred['shift_{}'.format(s)] = x_pred.iloc[:, 0].shift(s)
#         #     x_train['shift_{}'.format(s)] = x_train.iloc[:, 0].shift(s)
#         #
#         # x_pred = x_pred.dropna().values
#         # x_train = x_train.dropna().values
#         # y_train = y_train.dropna().iloc[:-ROLLING_WINDOW, ].values
#         x_pred = x_pred[:, None]
#         y_pred = createModelPrediction(x_train, y_train, x_pred)
#         y_pred = scaler.inverse_transform(y_pred)
#
#         y_pred = y_pred.flatten()
#
#         allData.addRawSheet(fileName, y_pred, dates)
#
#     externalData.createDataReport(allData, 'predict')

# Creates a report analyzing the model with a given split


def createAnalyzeProject(paths, split):
    dataHolder = DatumHolder()

    for path in paths:
        fileName = os.path.basename(path)
        csvData = dataEditing.readData(path)
        dates = csvData.index[-split:]
        allData = csvData.values

        y_pred, y_true = createModelPrediction(allData, split)
        dataHolder.addAnalyzeSheet(fileName, y_pred.flatten(), y_true.flatten(), dates)

    externalData.createDataReport(dataHolder, 'analyze')
