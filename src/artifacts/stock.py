# Install dependencies
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from datetime import datetime

# Reading in data from CSV
df = pd.read_csv('../data/WIKI_FB.csv')


graphed_data = pd.DataFrame(data=df[['Adj. Close']].values, index=df['Date'])
print(df[['Adj. Close']])
print(graphed_data)

df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = (df.Date - df.Date.min()).dt.days
df = df[['Date', 'Adj. Close']]



forecast_out = 180

x = df['Date'].values.reshape(-1, 1)
y = df['Adj. Close'].values

print(x, y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

print('Starting')
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_rbf.fit(x_train, y_train)
svm_confidence = svr_rbf.score(x_test, y_test)
print("svm confidence: ", svm_confidence)

# Train the model


# Testing Model: Score returns the coefficient of determination R^2 of the prediction.
# The best possible score is 1.0

last_date = df.Date.max()+1
print(last_date)
predict_x = np.arange(last_date, last_date+forecast_out).reshape(-1, 1)

svm_prediction = svr_rbf.predict(predict_x)
# print(svm_prediction)

print(svm_prediction)
svm_slope = svr_rbf.predict(df.Date.values.reshape(-1, 1))
svm_total_slope = svr_rbf.predict(np.arange(df.Date.min(), predict_x.max()).reshape(-1, 1))
fig, ax = plt.subplots(nrows=3)
ax[0].plot(df['Adj. Close'])
ax[0].plot(svm_slope, '--')
ax[1].plot(svm_prediction)
ax[2].plot(svm_total_slope)
plt.show()
