# Install dependencies
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from joblib import dump, load

def createReport():
    months = ['First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth']
    data_report = pd.DataFrame(
        columns=['Ticker', *[''.join([month, ' Month']) for month in months]])
    return data_report


def createSeries(acc, file_name):

    data = [file_name[:file_name.index('.')]]
    for prices in acc:
        average_pred = np.average(prices[0])
        average_accpt = np.average(prices[1])
        percent_error = (abs(average_accpt - average_pred) /
                         average_accpt) * 100
        data.append(percent_error)

    return data


# Reading in data from CSV
file_name = 'WIKI_COKE.csv'
df = pd.read_csv(''.join(['./data/', file_name]), index_col='Date')
# Splitting data into training and testing with 0.3 split
df.index = pd.to_datetime(df.index)
# Normalizing data
actual_values = df.tail(180)

forecast_out = 180
df = df[['Adj. Close']]
df['Prediction'] = df[['Adj. Close']].shift(-forecast_out)

x = np.array(df.drop(['Prediction'], 1))
x = x[:-forecast_out]

y = np.array(df['Prediction'])
y = y[:-forecast_out]

print('Starting')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_rbf.fit(x_train, y_train)
svm_confidence = svr_rbf.score(x_test, y_test)
print("svm confidence: ", svm_confidence)

lr = LinearRegression()
# Train the model
lr.fit(x_train, y_train)


# Testing Model: Score returns the coefficient of determination R^2 of the prediction.
# The best possible score is 1.0
lr_confidence = lr.score(x_test, y_test)
print("lr confidence: ", lr_confidence)

x_forecast = np.array(df.drop(['Prediction'], 1))[-forecast_out:]

lr_acc, svm_acc = [], []
lr_prediction = lr.predict(x_forecast)
# print(lr_prediction)# Print support vector regressor model predictions for the next '30' days
svm_prediction = svr_rbf.predict(x_forecast)
# print(svm_prediction)

for n in range(0, 180, 30):
    lr_acc.append([lr_prediction[n:n + 30],
                   actual_values.iloc[n:n + 30][['Adj. Close']].values])
    svm_acc.append([svm_prediction[n:n + 30],
                    actual_values.iloc[n:n + 30][['Adj. Close']].values])


data_report = createReport()
lr_series = createSeries(lr_acc, file_name)
svm_series = createSeries(svm_acc, file_name)

print(lr_series)
print(svm_series)
data_report.loc[len(data_report)] = lr_series
data_report.loc[len(data_report)] = svm_series
print(data_report)

data_report.to_csv("Six_month_report.csv", index=False)



# Plotting data

# print(lr.coef_, lr.intercept_)
# fig, ax = plt.subplots()
# ax.plot(pd.DataFrame(data=lr_prediction, index=df.index))
# plt.show()
