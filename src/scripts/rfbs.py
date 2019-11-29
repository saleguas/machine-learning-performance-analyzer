# Install dependencies
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn.externals import joblib

def createModel(data, filename):
    print('Modifying data')
    features = data[['Date', 'Adj. Close']].copy()
    features['Date'] = pd.to_datetime(features['Date'])
    features['Year'] = features['Date'].apply(lambda x: x.year)
    features['Month'] = features['Date'].apply(lambda x: x.month)
    features['Day'] = features['Date'].apply(lambda x: x.day)
    features = features.drop('Date', 1)
    features = pd.get_dummies(features, columns=['Year', 'Month', 'Day'])
    labels = np.array(features['Adj. Close'])
    features = features.drop('Adj. Close', 1)
    features = np.array(features)

    print('Training Regressor...')
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    rf.fit(features, labels);

    print('Saving {}.model: '.format(filename))
    filehandler = open('../models/{}.model'.format(filename), 'wb+')
    joblib.dump(rf, filehandler)
    filehandler.close()


def analyze(data, days):

    createModel(data, )


def predict(model, days, data):
    loaded_model = joblib.load(model)
    futureValues = {'Date' : pd.date_range(start=df[['Date']].iloc[-1].iat[0], periods=days)}
    df = pd.DataFrame(futureValues)
    df['Year'] = df['Date'].apply(lambda x: x.year)
    df['Month'] = df['Date'].apply(lambda x: x.month)
    df['Day'] = df['Date'].apply(lambda x: x.day)
    df = df.drop('Date',1)
    df = pd.get_dummies(df,columns=['Year', 'Month','Day'])
    predictedValues = loaded_model.predict(df)
    print(predictedValues)
# data is '../data/WIKI_CLX.csv'


createModel(pd.read_csv('../data/WIKI_CLX.csv'), 'teste')
predict("../models/test.model",30, "2018-01-01")


#for predict and analyze
# future_vals = pd.date_range(start=df[['Date']].iloc[-1].iat[0], periods=30)

#one hot



# train_features, test_features, train_labels, test_labels = train_test_split(
#     features, labels, test_size=0.25, random_state=42)
#
# baseline_preds = labels.mean()
# baseline_errors = abs(baseline_preds - test_labels)



# predictions = rf.predict(test_features)
# errors = abs(predictions - test_labels)
#
#
# rf_slope = rf.predict(features)
