import pandas as pd
import numpy as np


# Reads in the csv file with the given path. If split is 0, then it's assumed not to be analyzing and instead predicting future values. Returns pandas dataframe with date and close value
def readData(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    print(df)
    df = df.iloc[:, 3:4]
    return df

# Splits the time series data at a given point and uses getPredictData on the left half and returns the right half as test data. Returns the left half, the x_train, the y_train, the x_pred, the right half, and the dates


def getAnalyzeData(df, split):
    train = df.iloc[:-split]
    y_true = df.iloc[-split:]

    x_train, y_train, x_pred, dates = getPredictData(train, split)

    return train.values.reshape(-1, 1), x_train.values.reshape(-1, 1), y_train.values.reshape(-1, 1), x_pred.values.reshape(-1, 1), y_true.values.reshape(-1, 1), dates

# Shifts and edits the data and returns the x train, y train, the prediction input for the future values, and the dates of the future values as pandas series


def getPredictData(df, future):
    train = df.iloc[:]
    dates = pd.date_range(df.index[-1], periods=future)
    x_pred = train[-future:]
    x_train = train[:-future]
    y_train = train[future:]

    return x_train, y_train, x_pred, dates


def preprocessData(x_train_raw, y_train_raw, x_pred):
    x_train = []
    y_train = []

    for i in range(30, len(x_train_raw)):
        x_train.append(x_train_raw[i-30, 0])
        y_train.append(y_train_raw[i-30, 0])

    x_train = np.array(x_train).reshape(-1, 1)
    y_train = np.array(y_train).reshape(-1, 1)
    x_pred = np.array(x_pred).reshape(-1, 1)
    print(x_pred.shape)
    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    y_train = np.reshape(y_train, (y_train.shape[0], 1, y_train.shape[1]))
    x_pred = np.reshape(x_pred, (x_pred.shape[0], 1, x_pred.shape[1]))
    return x_train, y_train, x_pred


# arr = 1 2 3 4 5
# future = 1

# x_pred = 5
# x_train = 1 2 3 4
# y_train = 2 3 4 5
