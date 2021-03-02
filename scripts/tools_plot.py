import pandas as pd
import datetime
import matplotlib.pyplot as plt
import os


def plotGraph(df: pd.DataFrame, category: str, dataKey: str, saveFigDirectoryPath=None, verbose=False, startYear=2000,endYear=2050, dropNA=False) -> plt:
    data = df[["sum",dataKey]]

    start = data.index.searchsorted(datetime.datetime(startYear, 1, 1))
    end = data.index.searchsorted(datetime.datetime(endYear, 12, 31))
    data = data.iloc[start:end]

    if dropNA:
        data = data[data[dataKey] != "na"]

    print(data.groupby(dataKey)[dataKey].agg("count"))
    plt.clf()

    data.groupby(dataKey)['sum'].plot(stacked=True)
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    plt.title(category)
    plt.tight_layout()

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