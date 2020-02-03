import numpy as np
import pandas as pd


class Datum:

    def __init__(self, name, data):
        self.name = name
        self.data = data


class DatumHolder:

    def __init__(self):
        self.rawSheets = []
        self.analyzeSheets = []

    def addRawSheet(self, name, y_pred, dates):
        sheet = pd.DataFrame()
        sheet['Predicted Close'] = y_pred
        sheet['Date'] = dates
        datumSheet = Datum(name, sheet)
        self.rawSheets.append(datumSheet)

    def addAnalyzeSheet(self, name, y_pred, y_true, dates):
        sheet = pd.DataFrame()
        sheet['Date'] = dates
        sheet['Predicted Close'] = y_pred
        sheet['Actual Close'] = y_true
        sheet['relative % error'] = y_true - y_pred
        sheet['absolute % error'] = abs(y_true - y_pred) / y_pred * 100
        print(sheet.head())
        datumSheet = Datum(name, sheet)
        self.analyzeSheets.append(datumSheet)

    def getRawSheets(self):
        for datum in self.rawSheets:
            yield datum.name, datum.data

    def getAnalyzeSheets(self):
        for datum in self.analyzeSheets:
            yield datum.name, datum.data

    def generateAnalyzeReport(self):
        sheet = pd.DataFrame()
        names = [datum.name for datum in self.analyzeSheets]
        datesFrom = [datum.data['Date'].iloc[0]
                     for datum in self.analyzeSheets]
        datesTo = [datum.data['Date'].iloc[-1] for datum in self.analyzeSheets]
        averagePredictedClose = [
            datum.data['Predicted Close'].mean() for datum in self.analyzeSheets]
        averageActualClose = [datum.data['Actual Close'].mean()
                              for datum in self.analyzeSheets]
        averageRelError = [datum.data['relative % error'].mean()
                           for datum in self.analyzeSheets]
        averageAbsError = [datum.data['absolute % error'].mean()
                           for datum in self.analyzeSheets]
        sheet['Name'] = names
        sheet['Starting Date'] = datesFrom
        sheet['Ending Date'] = datesTo
        sheet['Average Predicted Close'] = averagePredictedClose
        sheet['Average Actual Close'] = averageActualClose
        sheet['Average Percent Relative Error'] = averageRelError
        sheet['Average Percent Absolute Error'] = averageAbsError
        return sheet

    def reportPossible(self):
        return len(self.analyzeSheets) != 0
