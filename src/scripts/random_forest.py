# Install dependencies
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVR
import objects

# Reading in data from CSV

def analyze(path, days, name):

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
    # features = pd.get_dummies(features, columns=['Year', 'Month', 'Day'])
    # Turning features into one hot encoding
    labels = np.array(features['Adj. Close'])
    # Gettig our labels
    features = features.drop('Adj. Close', 1)
    features = np.array(features)

    # Finished cleaning up data, print it to make sure
    print(features)
    print(labels)

    # parameters = {
    # "kernel": ["rbf"],
    # "C": [1,10,10,100,1000],
    # "gamma": [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    # }
    # param_grid = {'C': [0.1, 1, 10, 100, 1000],
    #           'gamma': [1, .9, .8, .7, .6, .5, .4, .3, .2, .1],
    #           'kernel': ['rbf']}
    # grid = GridSearchCV(SVR(), param_grid, cv=5, verbose=2)
    # grid.fit(features[:-days], labels[:-days])
    # print(grid.best_estimator_)
    # print(grid.best_params_)
    clf = SVR(C=1000, gamma=.55)
    clf.fit(features[:-days], labels[:-days])
    # clf.fit(features[:-days], labels[:-days])
    predicted_values = clf.predict(features[-days:])
    actual_values = labels[-days:]
    dates = df['Date'].iloc[-days:].values

    a = objects.analyzeFile(name, predicted_values, actual_values, dates)
    objects.createReport()
    # print(predicted_values)
    # print(actual_values)
    # fig, ax = plt.subplots()
    # ax.plot(predicted_values)
    # ax.plot(actual_values)
    # plt.show()

    # Splitting up our data into the training and testing data.
    # This is to analyze with a given amount of days
    # train_features = features[:-days]
    # train_labels = labels[:-days]
    # test_features = features[-days:]
    #
    # x_train, x_test, y_train, y_test = train_test_split(train_features, train_labels, test_size=.3)
    #
    #
    #
    # # Setting up the regressor and fitting the data
    # rf = RandomForestRegressor(n_estimators = 1000)
    # rf.fit(x_train, y_train)
    #
    # # Let's predict our test data now
    # predicted_values = rf.predict(x_test)
    # actual_values = y_test
    # dates = pd.to_datetime(df['Date']).iloc[-days:].values
    # print(dates)
    # print(predicted_values)
    # print(actual_values)

    # a = objects.analyzeFile(name, predicted_values, actual_values, dates)





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
analyze('../data/WIKI_AAL.csv', 60, 'stock')
