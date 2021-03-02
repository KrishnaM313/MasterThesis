
import pandas as pd
import matplotlib.pyplot as plt
import os
from tools_parties import fixParty, getPartyIdeology
from tqdm import tqdm
from tools_plot import plotGraph

if __name__ == '__main__':

    baseDir = "/home/user/workspaces/MasterThesis/data"
    analysisSummaryDir = os.path.join(baseDir,"analysisSummary")

    for category in ["climate", "health"]:
        
        fileAnalysisPath = os.path.join(analysisSummaryDir, "{category}.csv".format(category=category))
        df = pd.read_csv(fileAnalysisPath)
        df["sum"] = df.drop(["politicalGroup","date"],axis=1).sum(axis=1)
        data = df[["politicalGroup","sum","date"]]

        for i, entry in enumerate(tqdm(df["politicalGroup"])):
            update = fixParty(entry)
            df.loc[i, ["fixedPoliticalGroup"]] = update
            df.loc[i, ["politicalGroupIdeology"]] = getPartyIdeology(update)
        
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)

        for key in ["fixedPoliticalGroup","politicalGroupIdeology"]:
            plotGraph(df.copy(), category, key, analysisSummaryDir, 
            startYear=2018, dropNA=False, verbose=True)
        
        
