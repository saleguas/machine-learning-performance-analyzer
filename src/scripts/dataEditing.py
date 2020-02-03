import pandas as pd


# Reads in the csv file with the given path. If split is 0, then it's assumed not to be analyzing and instead predicting future values. Returns pandas dataframe with date and close value
def readData(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    df = df['Close']
    return df

# Splits the time series data at a given point and uses getPredictData on the left half and returns the right half as test data. Returns the left half, the x_train, the y_train, the x_pred, the right half, and the dates


def getAnalyzeData(df, split):
    train = df.iloc[:-split]
    y_true = df.iloc[-split:]

    x_train, y_train, x_pred, dates = getPredictData(train, split)

    return train, x_train, y_train, x_pred, y_true, dates

# Shifts and edits the data and returns the x train, y train, the prediction input for the future values, and the dates of the future values as pandas series


def getPredictData(df, future):
    train = df.iloc[:]
    dates = pd.date_range(df.index[-1], periods=future)
    x_pred = train[-future:]
    x_train = train[:-future]
    y_train = train[future:]

    return x_train, y_train, x_pred, dates


# arr = 1 2 3 4 5
# future = 1

# x_pred = 5
# x_train = 1 2 3 4
# y_train = 2 3 4 5
