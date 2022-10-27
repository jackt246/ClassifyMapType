import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Plotter():
    def __init__(self, df):
        self.df = df
        self.ExpectedType = df['Expected Type']

    def ScatterHue(self, column1, column2, Huecolumn):
        sns.scatterplot(x='{}'.format(column1), y='{}'.format(column2), data=self.df, hue='{}'.format(Huecolumn))
        plt.legend(loc='lower right')
        plt.title(self.ExpectedType.iloc[1])
        plt.show()

    def CountBar(self, column):
        sns.countplot(x=self.df['{}'.format(column)])
        plt.title(self.ExpectedType.iloc[1])
        plt.show()

    def Boxplot(self, column1, column2):
        sns.boxplot(data=self.df, x=str(column1), y=str(column2))
        plt.title(self.ExpectedType.iloc[1])
        plt.show()

def ReturnSortedDf(df, column, condition):
    newdf = df[df[str(column)] == str(condition)]
    return newdf

def CreateEmdList(df):
    emddflist = df['Map'].tolist()
    emdlist = []
    for row in emddflist:
        emdid = row.strip('.map.gz')
        emdlist.append(emdid)

    return emdlist




SubTomo = pd.read_csv('results_Tomograms_tomo.csv')

#DataPlotting = Plotter(df=SubTomo)
#DataPlotting.Boxplot('Predicted Type', 'Prediction score %')
#DataPlotting.CountBar('Predicted Type')
#DataPlotting.CountBar('What is it')

IPET = ReturnSortedDf(SubTomo, 'What is it', 'IPET')
emdIdList = CreateEmdList(IPET)
with open("IPETList.txt", 'w') as file:
    for row in emdIdList:
        file.write(row + '\n')