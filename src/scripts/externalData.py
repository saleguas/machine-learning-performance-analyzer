from datetime import datetime
import os
import pandas as pd
import pandas_datareader.data as web

# Downloads the ticker given


def downloadData(ticker, path='../data/'):
    symbol = 'WIKI/{}'.format(ticker)
    df = web.DataReader(symbol, 'quandl', api_key="xiypVZFNu6XEvRsCne29")
    df.to_csv(os.path.join(path, '{}.csv'.format(ticker)))


# Returns the path to the report folder. The type parameter is appended to the folder name
def reportPath(type):
    dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    report_base_folder = os.path.join('..', 'reports')
    report_name = 'report_{}_{}'.format(type, dt_string)
    report_path = os.path.join(report_base_folder, report_name)
    if not os.path.exists(report_base_folder):
        os.makedirs(report_base_folder)

    return report_path


# Creates a report folder with the current data.
def createDataReport(allData, type):
    report_path = reportPath(type)
    sheets_path = os.path.join(report_path, 'sheets')
    rawSheets_path = os.path.join(sheets_path, 'raw')
    analyzeSheets_path = os.path.join(sheets_path, 'anayze')

    os.mkdir(report_path)
    os.mkdir(sheets_path)
    os.mkdir(rawSheets_path)

    for sheetName, sheet in allData.getRawSheets():
        sheet.to_csv(os.path.join(rawSheets_path,
                                  '{}_raw_data.csv'.format(sheetName)), index=False)

    if allData.reportPossible():
        os.mkdir(analyzeSheets_path)

        for sheetName, sheet in allData.getAnalyzeSheets():
            sheet.to_excel(os.path.join(
                analyzeSheets_path, '{}_analyzed_data.xlsx'.format(sheetName)), index=False)

        analyzeSheet = allData.generateAnalyzeReport()
        analyzeSheet.to_excel(os.path.join(
            sheets_path, 'all_data_analyzed.xlsx'), index=False)
        analyzeSheet.to_csv(os.path.join(
            sheets_path, 'all_data_analyzed.csv'), index=False)
