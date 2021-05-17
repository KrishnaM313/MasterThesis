import datetime
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools_counter import getCounterPercentile
from tools_data import saveJSON


def plotGraph(df: pd.DataFrame, category: str, dataKey: str, saveFigDirectoryPath=None, verbose=False, startYear=2000, endYear=2050, dropNA=False, title=True) -> plt:
    data = df[["sum", dataKey]]

    start = data.index.searchsorted(datetime.datetime(startYear, 1, 1))
    end = data.index.searchsorted(datetime.datetime(endYear, 12, 31))
    data = data.iloc[start:end]

    if dropNA:
        data = data[data[dataKey] != "na"]

    groups = data.groupby(dataKey)

    overview = groups[dataKey].agg("count")

    print(overview)
    if saveFigDirectoryPath is not None:
        overviewFilePath = os.path.join(saveFigDirectoryPath, "{category}_{dataKey}.json".format(
            category=category, dataKey=dataKey))
        overview["total"] = overview.values.sum()
        overview.to_json(overviewFilePath)

    plt.clf()

    groups['sum'].plot()
    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
               mode="expand", borderaxespad=0, ncol=2)

    plt.ylabel("count")

    if title:
        plt.title(category)
    plt.subplots_adjust(top=0.75)
    if saveFigDirectoryPath is not None:
        imgFilePath = os.path.join(saveFigDirectoryPath, "{category}_{dataKey}.pdf".format(
            category=category, dataKey=dataKey))
        plt.savefig(imgFilePath)

    return plt


def plotCounterHistogram(counter: Counter, category="text_length", saveFigDirectoryPath=None, saveJSONFile=True, showPlot=False, startYear=None, endYear=None, small=False, title=None):
    plt.clf()

    labels, values = zip(*counter.items())

    indexes = np.arange(len(labels))
    width = 1

    density = False
    if small:
        density = True

    align = None
    if small:
        align = "edge"

    plt.bar(indexes, values, width, align=align)

    percentile95 = getCounterPercentile(95, counter)
    plt.axvline(x=percentile95, color="red",
                label="95 percentile ("+str(percentile95)+")")

    if title is not None:
        title = "Speeches "+str.title(category)+" Count"
        if startYear is not None:
            title = title + " " + str(startYear) + "-" + str(endYear)
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
                saveJSON(dict(counter), os.path.join(
                    saveFigDirectoryPath, filename + ".json"))
