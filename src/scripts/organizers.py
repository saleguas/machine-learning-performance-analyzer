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
        print(y_pred.shape)
        print(y_true.shape)
        sheet = pd.DataFrame()
        sheet['Date'] = dates
        sheet['Days'] = np.arange(1, len(dates)+1)
        sheet['Predicted Close'] = y_pred
        sheet['Actual Close'] = y_true
        sheet['relative % error'] = (sheet['Predicted Close'] - sheet['Actual Close']) / sheet['Predicted Close'] * 100
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

    def generateForecastReport(self):
        sheet = pd.DataFrame()
        names = [datum.name for datum in self.analyzeSheets]
        oneDayRelError = [datum.data['relative % error'].iloc[0].mean() for datum in self.analyzeSheets]
        oneWeekRelError = [datum.data['relative % error'].iloc[:7].mean() for datum in self.analyzeSheets]
        oneMonthRelError = [datum.data['relative % error'].iloc[:30].mean() for datum in self.analyzeSheets]
        threeMonthRelError = [datum.data['relative % error'].iloc[:90].mean() for datum in self.analyzeSheets]
        sixMonthRelError = [datum.data['relative % error'].iloc[:180].mean() for datum in self.analyzeSheets]
        oneDayAbsError = [datum.data['absolute % error'].iloc[0].mean() for datum in self.analyzeSheets]
        oneWeekAbsError = [datum.data['absolute % error'].iloc[:7].mean() for datum in self.analyzeSheets]
        oneMonthAbsError = [datum.data['absolute % error'].iloc[:30].mean() for datum in self.analyzeSheets]
        threeMonthAbsError = [datum.data['absolute % error'].iloc[:90].mean() for datum in self.analyzeSheets]
        sixMonthAbsError = [datum.data['absolute % error'].iloc[:180].mean() for datum in self.analyzeSheets]

        sheet['Name'] = names

        sheet['1 day Rel'] = oneDayRelError
        sheet['1 day Abs'] = oneDayAbsError

        sheet['1 week Rel'] = oneWeekRelError
        sheet['1 week Abs'] = oneWeekAbsError

        sheet['1 month Rel'] = oneMonthRelError
        sheet['1 month Abs'] = oneMonthAbsError

        sheet['3 month Rel'] = threeMonthRelError
        sheet['3 month Abs'] = threeMonthAbsError

        sheet['6 month Rel'] = sixMonthRelError
        sheet['6 month Abs'] = sixMonthAbsError

        return sheet


    def reportPossible(self):
        return len(self.analyzeSheets) != 0

    def forecastPossible(self):
        return min([len(datum.data['relative % error']) for datum in self.analyzeSheets])
