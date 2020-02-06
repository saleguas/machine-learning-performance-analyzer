import argparse
import os
import modelCreation
import externalData
from gooey import Gooey

def getFilePaths(path):
    paths = []
    for root, dirs, files in os.walk(os.path.abspath(path)):
        for file in files:
            paths.append(os.path.join(root, file))
    return paths

@Gooey
def main():
    parser = argparse.ArgumentParser(
        description='Interface for the stock prediction models. Every file in the data folder will be processed with the given command and report(s) are generated.')

    parser.add_argument(
        '-d', '--data', help='Pass the Tickers of the stocks to download', metavar=("TICKER"), nargs='+')
    parser.add_argument('-i', '--input', help='Specify the input folder. By defult it is /data',
                        metavar=('LOCATION'), nargs=1)
    # parser.add_argument('-p', '--predict', help='Predict FUTURE values into the future for each stock',
    #                     metavar=('FUTURE'), nargs=1)
    parser.add_argument('-a', '--analyze', help='Analyze the efficiency, splitting the time series data at SPLIT',
                        metavar=('SPLIT'), nargs=1)
    args = vars(parser.parse_args())

    location = os.path.join('..', 'data')

    if args['data']:
        for arg in args['data']:
            externalData.downloadData(arg)

    if args['input']:
        location = args['input'][0]

    # if args['predict']:
    #     paths = getFilePaths(location)
    #     modelCreation.createPredictionProject(paths, int(args['predict'][0]))

    if args['analyze']:
        paths = getFilePaths(location)
        modelCreation.createAnalyzeProject(paths, int(args['analyze'][0]))

main()
