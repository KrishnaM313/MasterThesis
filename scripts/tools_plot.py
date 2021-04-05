import pandas as pd
import datetime
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.dates as dates
from collections import Counter
import matplotlib.pyplot as plt
from tools_data import saveJSON
from tools_counter import getCounterPercentile


def plotGraph(df: pd.DataFrame, category: str, dataKey: str, saveFigDirectoryPath=None, verbose=False, startYear=2000,endYear=2050, dropNA=False, title=True) -> plt:
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
        overview["total"] = overview.values.sum()
        overview.to_json(overviewFilePath)

    plt.clf()

    groups['sum'].plot(stacked=True)
    #plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    #plt.legend(loc=(1.04,0))
    plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2)

    if title:
        plt.title(category)
    #plt.tight_layout()


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
    plt.subplots_adjust(top=0.75)
    if saveFigDirectoryPath is not None:
        imgFilePath = os.path.join(saveFigDirectoryPath,"{category}_{dataKey}.pdf".format(category=category, dataKey=dataKey))
        plt.savefig(imgFilePath)
    
    return plt

def plotCounterHistogram(counter: Counter, category="text_length", saveFigDirectoryPath=None, saveJSONFile=True, showPlot=False, startYear=None, endYear=None, small=False):
    plt.clf()


    labels, values = zip(*counter.items())
    
    


    indexes = np.arange(len(labels))
    width = 1

    density = False
    if small:
        density = True

    align = None
    if small:
        align="edge"

    plt.bar(indexes, values, width, align=align)

    percentile95 = getCounterPercentile(95,counter)
    plt.axvline(x=percentile95, color="red", label="95 percentile ("+str(percentile95)+")")

    # mean, variance, std_dev = getCounterStats(counter)
    # plt.axvline(x=percentile95, color="green", label="mean")
    # plt.axvline(x=percentile95, color="yellow", label="median")

    title = "Speeches "+str.title(category)+" Count"
    if startYear is not None:
        title = title + " " + str(startYear) + "-" +  str(endYear)
    plt.title(title)
    plt.legend()

    if showPlot is True:
        plt.show()
    else:
        if saveFigDirectoryPath is not None:
            filename = "speeches_count_" + str.lower(category)
            if startYear is not None:
                filename = filename + "_" + str(startYear) + "_" + str(endYear)

            plotPath = os.path.join(saveFigDirectoryPath, filename)
            plt.savefig(plotPath+".png")
            plt.savefig(plotPath+".svg", bbox_inches='tight')
            plt.savefig(plotPath+".pdf", bbox_inches='tight')
            if saveJSONFile:
                saveJSON(dict(counter), os.path.join(saveFigDirectoryPath, filename + ".json"))

