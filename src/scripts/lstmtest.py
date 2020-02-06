import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from dataEditing import readData

allData = readData('../data/AMT.csv')
sc = MinMaxScaler(feature_range=(0, 1))
allData_scaled = sc.fit_transform(allData)
training_set_scaled = allData_scaled[:-180]
testing_set_scaled = allData_scaled[-180:]

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

##########################################################

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

##########################################################

y_pred = model.predict(x_pred)
y_pred = sc.inverse_transform(y_pred)
print(y_pred)
print(y_true)
print(y_pred.shape)
print(y_true.shape)
df = pd.DataFrame()
df['Actual'] = y_true.flatten()
df['Predicted'] = y_pred.flatten()
print()
df.plot()
plt.show()
