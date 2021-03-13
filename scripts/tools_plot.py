import pandas as pd
import datetime
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.dates as dates
from collections import Counter
import matplotlib.pyplot as plt
from tools_data import saveJSON


def plotGraph(df: pd.DataFrame, category: str, dataKey: str, saveFigDirectoryPath=None, verbose=False, startYear=2000,endYear=2050, dropNA=False) -> plt:
    data = df[["sum",dataKey]]

    start = data.index.searchsorted(datetime.datetime(startYear, 1, 1))
    end = data.index.searchsorted(datetime.datetime(endYear, 12, 31))
    data = data.iloc[start:end]

    if dropNA:
        data = data[data[dataKey] != "na"]

    groups = data.groupby(dataKey)

    overview = groups[dataKey].agg("count")
    print(overview)
    if saveFigDirectoryPath is not None:
        overviewFilePath = os.path.join(saveFigDirectoryPath,"{category}_{dataKey}.json".format(category=category, dataKey=dataKey))
        overview.to_json(overviewFilePath)

    plt.clf()

    groups['sum'].plot(stacked=True)
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    plt.title(category)
    plt.tight_layout()


    # print(data.head())
    # exit()
    # for memberEntries in groups:
    #     print(memberEntries[["date"]])
    #     x = dates.date2num(memberEntries[["date"]])
    #     y = memberEntries[["sum"]]

    #     z = np.polyfit(x, y, 1)
    #     p = np.poly1d(z)
    #     plt.plot(x,p(x),"r--")

    # data = data.groupby("politicalGroup")
    # print(data.head())

    # for i, (k, g) in enumerate(data):
    #     plt.plot_date(g['sum'].index, g['sum'], linestyle='None', marker='o', label=k)

    # plt.legend()
    #plt.plot(data)
    #df[['climate change','global warming']].sum().plot.bar()
    #print(df.head())
    if saveFigDirectoryPath is not None:
        imgFilePath = os.path.join(saveFigDirectoryPath,"{category}_{dataKey}.png".format(category=category, dataKey=dataKey))
        plt.savefig(imgFilePath)
    
    return plt

def plotCounterHistogram(counter: Counter, category="text_length", saveFigDirectoryPath=None, saveJSONFile=True, showPlot=False):
    plt.clf()
    labels, values = zip(*counter.items())

    indexes = np.arange(len(labels))
    width = 1

    plt.bar(indexes, values, width)
    plt.title(str.title("Speeches "+category+" Count"))

    if showPlot is True:
        plt.show()

    if saveFigDirectoryPath is not None:
        plotPath = os.path.join(saveFigDirectoryPath, "speeches_count_" + str.lower(category) + ".png")
        plt.savefig(plotPath)
        if saveJSONFile:
            saveJSON(dict(counter), os.path.join(saveFigDirectoryPath, "speeches_count_" + str.lower(category) + ".json"))