# import the necessary packages
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
source_group = ap.add_argument_group(
    title='Source', description='Commands to collect data')
data_group = ap.add_argument_group(
    title='Data management', description='Commands to manage and organize data/results')


source_group.add_argument("-d", "--download",
                          help='''Download stock data from quandl. Pass in the name of the quandl location. Saves to the ./data directory. Only works for free databases. Example: --download WIKI/COKE''')
data_group.add_argument("-cfh", "--createFrameHolder",
                        help="Creates a blank frame analyzer holder object. Pass the name of the file. Saves to ./objects/holder directory. Example: --createFrameHolder bankingRecords")


args = vars(ap.parse_args())

from scripts.downloadData import downloadData
if args['download']:
    downloadData(args['download'])
# display a friendly message to the user
