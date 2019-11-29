# Install dependencies
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor


# Reading in data from CSV
df = pd.read_csv('./data/WIKI_FB.csv')
features = df[['Date', 'Adj. Close']].copy()
features['Date'] = pd.to_datetime(features['Date'])
features['Year'] = features['Date'].apply(lambda x: x.year)
features['Month'] = features['Date'].apply(lambda x: x.month)
features['Day'] = features['Date'].apply(lambda x: x.day)
features = features.drop('Date', 1)


print(features)
print(df[['Date']].iloc[-1].iat[0])
future_vals = pd.date_range(start=df[['Date']].iloc[-1].iat[0], periods=30)
print(future_vals)


features = pd.get_dummies(features, columns=['Year', 'Month', 'Day'])


labels = np.array(features['Adj. Close'])
features = features.drop('Adj. Close', 1)

features = np.array(features)
print(features)
print(labels)

train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.25, random_state=42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

baseline_preds = labels.mean()
baseline_errors = abs(baseline_preds - test_labels)
print('Average baseline error: ', round(np.mean(baseline_errors), 2))

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)

rf.fit(train_features, train_labels);

predictions = rf.predict(test_features)
errors = abs(predictions - test_labels)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

rf_slope = rf.predict(features)

print(rf_slope)
graphed_data = df[['Adj. Close']]
fig, ax = plt.subplots()
ax.plot(graphed_data)
ax.plot(rf_slope)
plt.show()
