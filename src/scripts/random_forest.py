# Install dependencies
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor


# Reading in data from CSV

def analyze(path, days):

    # Reading in data
    df = pd.read_csv(path)
    # Organizing data
    features = df[['Date', 'Adj. Close']].copy()
    features['Date'] = pd.to_datetime(features['Date'])
    features['Year'] = features['Date'].apply(lambda x: x.year)
    features['Month'] = features['Date'].apply(lambda x: x.month)
    features['Day'] = features['Date'].apply(lambda x: x.day)
    features = features.drop('Date', 1)
    print(features)
    features = pd.get_dummies(features, columns=['Year', 'Month', 'Day'])
    # Turning features into one hot encoding
    labels = np.array(features['Adj. Close'])
    # Gettig our labels
    features = features.drop('Adj. Close', 1)
    features = np.array(features)

    # Finished cleaning up data, print it to make sure
    print(features)
    print(labels)

    # Splitting up our data into the training and testing data.
    # This is to analyze with a given amount of days
    train_features = features[:-days]
    train_labels = labels[:-days]
    test_features = features[-days:]

    # Setting up the regressor and fitting the data
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    rf.fit(train_features, train_labels)

    # Let's predict our test data now
    predicted_values = rf.predict(test_features)
    actual_values = labels[-days:]
    dates = pd.to_datetime(df['Date']).iloc[-days:].values
    print(dates)
    print(predicted_values)
    print(actual_values)

    fig, ax = plt.subplots()
    ax.plot(predicted_values)
    ax.plot(actual_values)
    plt.show()



    # OLD CODE #

    # predictions = rf.predict(features)
    # baseline_preds = labels.mean()
    # baseline_errors = abs(baseline_preds - labels)
    # errors = abs(predictions - labels)
    # print('Average baseline error: ', round(np.mean(baseline_errors), 2))
    # print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

    # Graphing the data
    # rf_slope = rf.predict(features)
    # print(rf_slope)
    # graphed_data = df[['Adj. Close']]
    # fig, ax = plt.subplots()
    # ax.plot(graphed_data)
    # ax.plot(rf_slope)
    # plt.show()

def predict(path, days):
    pass

    # To be implemented

    # print(df[['Date']].iloc[-1].iat[0])
    # future_vals = pd.date_range(start=df[['Date']].iloc[-1].iat[0], periods=30)
    # print(future_vals)
analyze('../data/WIKI_AAL.csv', 180)
