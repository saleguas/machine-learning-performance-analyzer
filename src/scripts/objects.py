class DataFrameHolder():

    def __init__(self, name):
        self.name = name
        self.files = []

    def addFrame(self, *args):
        for file in args:
            self.files.append(file)

    def extractFrame(self, *args):
        for file in args:
            filePointer = self.files.index(file)
            exFile = self.files.pop(filePointer)
            print(exFile)



class frameAnalyzer():

    def __init__(self, name, data):
        self.name = name
        self.data = data


dfh = DataFrameHolder("hello")
dfh.addFrame("a", "b", "c")
print(dfh.files)
dfh.extractFrame("b")
