
import pandas as pd
import matplotlib.pyplot as plt
import os
from tools_parties import fixParty
from tqdm import tqdm

if __name__ == '__main__':

    baseDir = "/home/user/workspaces/MasterThesis/data"
    analysisSummaryDir = os.path.join(baseDir,"analysisSummary")

    for category in ["climate", "health"]:
        
        fileAnalysisPath = os.path.join(analysisSummaryDir, "{category}.csv".format(category=category))
        df = pd.read_csv(fileAnalysisPath)
        df["sum"] = df.drop(["politicalGroup","date"],axis=1).sum(axis=1)
        data = df[["politicalGroup","sum"]]
        #data.index = df["date"]

        

        for i, entry in enumerate(tqdm(df["politicalGroup"])):
            df.loc[i, ["politicalGroup"]] = fixParty(entry)
        
        print(df.groupby('politicalGroup')["politicalGroup"].agg("count"))
        

        df["date"] = pd.to_datetime(df["date"])

        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        df.groupby('politicalGroup')['sum'].plot(stacked=True)
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
        imgFilePath = os.path.join(analysisSummaryDir,"{category}.png".format(category=category))
        plt.savefig(imgFilePath)
        plt.clf()
        

