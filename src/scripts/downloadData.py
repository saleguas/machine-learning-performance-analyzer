import quandl
import os


def downloadData(ticker):
    print("Downloading {}...".format(ticker))
    df = quandl.get(ticker)
    filename = ''.join([ticker, '.csv']).replace('/', '_')
    df.to_csv(os.path.join('.', 'data', filename))
