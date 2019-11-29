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
data_group.add_argument("-cfa", "--createFrameAnalyzer",
                        help="")
data_group.add_argument("-af", "--addFrame", nargs='*',
                        help="Adds a data file to a frame holder. Pass the frame holder first, followed by the files to add. Example: --addFrame bankingRecords.dfh file1.anlyz file2.pred file3.anal...")
data_group.add_argument("-ef", "--extractFrame", nargs='*',
                        help="Extracts frame from a frame holder object. Pass the frame holder first, followed by the files to extract. Example: --extractFrame bankingRecords.dfh file1.data file2.data file3.data...")
data_group.add_argument("-lf", "--listFrame",
                        help="Lists all the data files in a frame holder object. Pass the name of the frame holder. Example: --listFrame bankingRecords.dfg")

args = vars(ap.parse_args())

from scripts.downloadData import downloadData
if args['download']:
    downloadData(args['download'])
# display a friendly message to the user
