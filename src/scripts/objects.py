import pandas as pd
import numpy as np
import os

class analyzeFile():

    def __init__(self, name, predicted, actual, dates):
        self.name = name
        self.df = pd.DataFrame()
        self.df['Date'] = dates
        self.df['Predicted'] = predicted
        self.df['Actual'] = actual
        print(self.df)
        self.addError()
        self.save()

    def addError(self):
        error = (self.df['Predicted']-self.df['Actual'])/self.df['Actual']*100
        self.df['Percent Error'] = error
        print(self.df)

    def getStats(self):
        baseline_preds = self.df['Actual'].mean()
        baseline_error = np.mean(abs(baseline_preds - self.df['Actual']))
        actual_error = np.mean(abs(self.df['Predicted'] - self.df['Actual']))
        print(baseline_error)
        print(actual_error)

    def save(self):
        self.df.to_pickle('../objects/{}.pkl'.format(self.name))

def createReport():
    files = os.listdir('../objects')
    print(files)
